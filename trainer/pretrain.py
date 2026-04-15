import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_utils import set_seed, load_ckp, init_model, SkipBatchSampler, Logger, get_lr, is_main_process, init_distributed_mode
from model.minimind import MinimindConfig
import argparse
import torch
from contextlib import nullcontext
from dataset.datasets import PretrainDataset
from torch import optim, nn
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def train_epoch(epoch, dataloader, iters, wandb, start_step=0):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(dataloader, start=start_step+1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        last_step = step

        with autocast_ctx:
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss + output.aux_loss
            loss /= args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.update(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            cur_aux_loss = output.aux_loss.item() if output.aux_loss is not None else 0.0
            cur_logits_loss = cur_loss - cur_aux_loss
            cur_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch[{epoch+1}/{args.epochs}], step[{step}/{iters}]: loss: {cur_loss:.4f}, aux_loss:{cur_aux_loss:.4f}, logits_loss:{cur_logits_loss:.4f}, lr:{cur_lr:.8f}, eta_min: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": cur_loss, "logits_loss": cur_logits_loss, "aux_loss": cur_aux_loss, "learning_rate": cur_lr, "epoch_time": eta_min})

        # checkpoint保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_path = '_moe' if mm_config.use_moe else ''
            ckp_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_path}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            load_ckp(lm_config=mm_config, epoch=epoch, step=step, model=model, optimizer=optimizer, prefix=args.save_weight, save_dir=args.save_dir, use_moe=args.use_moe)
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp_path)
            model.train()
            del state_dict
    
        del input_ids, labels, output, loss
    # 最后几个batch没更新参数
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.update(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

# 预训练
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", default=True, action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 1. 初始化随机数种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f'cuda:{local_rank}'
    set_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. 模型配置，ckp检查
    os.makedirs(args.save_dir, exist_ok=True)
    mm_config = MinimindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = load_ckp(lm_config=mm_config, prefix=args.save_weight, save_dir='../checkpoints', use_moe=mm_config.use_moe) if args.from_resume==1 else None

    # 3. 混合精度设置
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 训练可视化
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 模型、数据、优化器初始化
    model, tokenizer = init_model(lm_config=mm_config, from_weight=args.from_weight, device=args.device)
    train_ds = PretrainDataset(data_path=args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. 从ckp处恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model = ckp_data['model']
        scaler = ckp_data['scaler']
        optimizer = ckp_data['scaler']
        start_epoch = ckp_data['start_epoch']
        start_step = ckp_data.get('step', 0)

    # 7. 编译 + 分布式包装
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        set_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        dataloader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'训练跳过{skip}个batch, 下面从第{start_step + 1}个开始训练')
            train_epoch(epoch, dataloader, len(dataloader) + skip, wandb=wandb, start_step=start_step)
        else:
            train_epoch(epoch, dataloader, len(dataloader), wandb=wandb, start_step=start_step)

    # 9. 清理分布式进程
    if dist.is_initialized(): dist.destroy_process_group()