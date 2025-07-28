import argparse
import os 
import json
import datetime
import tqdm

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP   
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler    
from torch.utils.tensorboard import SummaryWriter        
import torch.distributed as dist                      

from vggt.models.vggt import VGGT
from vggt.finetuning.vggt_model_lora import LoRAVGGT
from vggt.finetuning.vggt_dataset_raw import VGGTDatasetRaw

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("-o", "--save_dir", type=str, required=True)
    parser.add_argument("--train_image_root", type=str, default="/scratch/ondemand28/ykguo/data_lowlight_new/processed_A1_0405/")
    parser.add_argument("--resume_weights", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--losses", nargs='+', default=["aggr_tokens_last"])
    parser.add_argument("--use_clean_jpg", action="store_true", help="Use clean jpg instead of clean npy for training")
    # Losses - determines scaling 
    parser.add_argument("--loss_scale_aggregator_tokens", type=float, default=1)
    # LoRA params 
    ### Aggregator 
    parser.add_argument("--lora_rank_aggr_patch_embed_qkv", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_patch_embed_qkv", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_patch_embed_proj", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_patch_embed_proj", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_patch_embed_mlp", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_patch_embed_mlp", type=int, default=16) 
    parser.add_argument("--lora_rank_aggr_frame_blocks_qkv", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_frame_blocks_qkv", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_frame_blocks_proj", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_frame_blocks_proj", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_frame_blocks_mlp", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_patch_embed_mlp", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_global_blocks_qkv", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_global_blocks_qkv", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_global_blocks_proj", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_global_blocks_proj", type=int, default=16)
    parser.add_argument("--lora_rank_aggr_global_blocks_mlp", type=int, default=16)
    parser.add_argument("--lora_alpha_aggr_global_blocks_mlp", type=int, default=16)
    ### 
    parser.add_argument("--scaling_constant_range", type=float, nargs=2, default=(18.0, 20.0))
    parser.add_argument("--sequence_length_range", type=int, nargs=2, default=(10, 15))
    parser.add_argument("--sampling_rate", type=float, default=3) 
    parser.add_argument("--noise_alphas", type=float, nargs=3, default=[5.96, 3.13, 6.81])
    parser.add_argument("--noise_betas", type=float, nargs=3, default=[-3669, -1991, -4189])
    parser.add_argument("--lr_starting", type=float, default=0.001) 
    parser.add_argument("--lr_min", type=float, default=0.0) 
    parser.add_argument("--lr_gamma", type=float, default=0.97)
    parser.add_argument("--use_distorted", action="store_true")
    parser.add_argument("--distortion_coefficents", type=str, default=None)
    args = parser.parse_args()

    # ============================
    # DDP: Initialize distributed process group and set device based on local_rank.
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)  # Set the GPU for this process first.
    dist.init_process_group("gloo")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()  # Get the rank of the current process.
    print(f"Process rank {dist.get_rank()}, local_rank {local_rank}, device {torch.cuda.current_device()}")
    # ============================

    now = datetime.datetime.now()
    # Only rank 0 handles directory creation and logging.
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        weights_dir = os.path.join(args.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        if not args.overwrite and os.listdir(weights_dir):
            raise ValueError(f"Directory {args.save_dir}/weights already exists and is not empty. Use --overwrite to overwrite.")
        figs_dir = os.path.join(args.save_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

    # Ensure all processes wait until rank 0 has created directories.
    dist.barrier()

    # TODO: configure this dataset init properly 
    # train 
    train_dataset = VGGTDatasetRaw(
        root_path=args.train_image_root,
        scaling_constant_range=args.scaling_constant_range,
        sampling_rate=args.sampling_rate,
    )

    # ============================
    # DDP: Use DistributedSampler so each process gets a subset of the data.
    train_sampler = DistributedSampler(train_dataset)         
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)  # [Modified]
    # ============================

    steps_per_epoch = len(train_loader)
    scheduler_step = int(steps_per_epoch / 10)

    original_model = VGGT().to(device)
    original_model.load_state_dict(torch.load("downloads/model.pt", map_location=device))

    for param in original_model.parameters():    
        param.requires_grad = False
    
    model = LoRAVGGT().to(device)
    model.load_state_dict(torch.load("downloads/model.pt", map_location=device))

    if args.resume_weights and os.path.exists(args.resume_weights):
        model.load_pretrained_lora(args.resume_weights)
        starting_epoch = int(args.resume_weights.split("/")[-1].split("_")[0]) + 1
        step = starting_epoch * steps_per_epoch
    else:
        model.setup_lora_model(args)
        starting_epoch = 0 
        step = -1 

    # ============================
    # DDP: Wrap the model with DistributedDataParallel.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  
    # ============================

    trainable_params = []
    for param in model.parameters():    
        if param.requires_grad:
            trainable_params.append(param)
    if rank == 0:
        # print all of the names of the trainable parameters
        print("***\nTrainable parameters:\n***")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
        print(f"===\nTotal number of trainable params: {len(trainable_params)}\n===")

        print(f"steps per epoch: {steps_per_epoch} | scheduler_step: {scheduler_step}")

    optimizer = optim.Adam(trainable_params, lr=args.lr_starting) # learning rate decay? 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_gamma)
    
    if rank == 0:
        args_out = vars(args)
        args_out["num_trainable_params"] = len(trainable_params)
        with open(os.path.join(args.save_dir, "lora_finetuning_args.json"), "w") as f:
            json.dump(args_out, f, indent=4)

    # ============================
    if rank == 0:
        writer = SummaryWriter(log_dir=args.save_dir)                   
    else:
        writer = None                                                   
    # ============================

    supervise_clean = any(loss.endswith("_clean") for loss in args.losses)  # Check if any loss requires clean supervision

    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        train_loss = 0
        # DDP: Set epoch for sampler for proper shuffling.
        train_sampler.set_epoch(epoch)                                      
        progress_bar = tqdm.tqdm(train_loader, desc=f"Training Progress (Epoch {epoch+1})", disable=(rank != 0))  # [Modified]

        for clean_sequence, noisy_sequence in progress_bar:
            step += 1
            loss = torch.tensor(0.0).to(device)
            loss_string = ""

            clean_aggregated_tokens = original_model.aggregator(clean_sequence)
            noisy_aggregated_tokens = model.module.aggregator(noisy_sequence)

            if supervise_clean: 
                print(f"Warning, clean supervision not yet implemented. ")

            if "aggr_tokens_last" in args.losses:
                aggr_tokens_last_loss = ((clean_aggregated_tokens[-1] - noisy_aggregated_tokens[-1])**2).mean()

                aggr_tokens_last_loss *= args.loss_scale_aggregator_tokens
                loss += aggr_tokens_last_loss

                loss_string += f"AgT: {aggr_tokens_last_loss.item():.2f} "
                if writer is not None:
                    writer.add_scalar("Loss/aggr_tokens_last_loss", aggr_tokens_last_loss.item(), step)


            loss_string = f"Tot: {loss.item():.3f} (lr: {optimizer.param_groups[0]['lr']:.6f}) | " + loss_string
            progress_bar.set_description(loss_string)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalar("Loss/Train", loss.item(), step)
            train_loss += loss.item()

            if step % scheduler_step == 0 and step > 0:
                if optimizer.param_groups[0]['lr'] > args.lr_min:
                    scheduler.step()
                if writer is not None:
                    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], step)

        if rank == 0:
            model.module.save_lora_weights(os.path.join(weights_dir, f"{str(epoch).zfill(4)}_lora.pth"))
            print(f"Epoch {epoch+1}/{args.num_epochs} | Avg Train Loss: {(train_loss / steps_per_epoch):.3f} ")
            writer.add_scalar("Loss/Train_epoch_avg", (train_loss / steps_per_epoch), epoch)

    if writer is not None:
        writer.close()
    
    # ============================
    # DDP: Clean up the process group.
    dist.destroy_process_group()                                
    # ============================
