import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_utils import set_seed, load_ckp, init_model, SkipBatchSampler, Logger, get_lr, is_main_process, init_distributed_mode
from model.minimind import MinimindConfig
import argparse
import torch
from contextlib import nullcontext
from dataset.datasets import SFTDataset
from torch import optim, nn
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def train_epoch(dataloader, epoch=0, iters=0, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_id, label) in enumerate(dataloader, start=start_step+1):
        input_id = input_id.to(args.device)
        label = label.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            output = model(input_ids=input_id, labels=label)
            loss = output.loss + output.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            cur_aux_loss = output.aux_loss.item() if output.aux_loss is not None else 0.0
            cur_logits_loss = cur_loss - cur_aux_loss
            cur_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(1, step - start_step) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {cur_loss:.4f}, logits_loss: {cur_logits_loss:.4f}, aux_loss: {cur_aux_loss:.4f}, lr: {cur_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": cur_loss, "logits_loss": cur_logits_loss, "aux_loss": cur_aux_loss, "learning_rate": cur_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_path}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = hasattr(raw_model, '_orig_mod', raw_model)
            state_dict = model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            load_ckp(lm_config=lm_config, epoch=epoch, step=step, model=model, optimizer=optimizer, 
                    wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_id, label, output, loss
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 1. 初始化随机种子 + 分布式初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f'cuda:{local_rank}'
    set_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. 模型配置初始化
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MinimindConfig(args.hidden_size, args.num_hidden_layers, bool(args.use_moe), args)
    ckp_data = load_ckp(lm_config=lm_config, save_dir=args.save_dir, prefix=args.save_weight)

    # 3. 设置混合精度
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.amp.cuda.autocast(dtype=dtype)

    # 4. 日志记录
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 初始化模型、tokenizer、数据
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight, save_dir=args.save_dir, device=device_type)
    train_ds = SFTDataset(data_path=args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. 从ckp处恢复数据
    start_epoch, start_step = 0, 0
    if ckp_data:
        model_weight = ckp_data['model']
        optimizer = ckp_data['optimizer']
        scaler = ckp_data['scaler']
        start_epoch = ckp_data['epoch']
        start_step = ckp_data['step']

    # 7. 编译、分布式包装
    if args.use_compile:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        set_seed(42 + epoch)
        skip = start_step if (start_epoch == epoch and start_step) else 0
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        dataloader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch[{epoch+1}/{args.epochs}]，跳过前{start_step}个batch进行训练')
            train_epoch(dataloader=dataloader, epoch=epoch, iters=len(dataloader) + skip, start_step=skip, wandb=wandb)
        else:
            train_epoch(dataloader=dataloader, epoch=epoch, iters=len(dataloader), start_step=0, wandb=wandb)

    # 9. 清理分布式进程
    if dist.is_initialized(): dist.destroy_process_group()