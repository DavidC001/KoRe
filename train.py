from KG_LM.configuration import load_yaml_config
from KG_LM.trainer import KG_LM_Trainer
import argparse
import logging
import wandb

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train KG-LFM model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to configuration file.")
    parser.add_argument("--time_budget", type=int, default=None, help="Time budget for training in seconds.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for training.")
    
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    debug_mode = args.debug
    
    logging.basicConfig(
        level=logging.INFO if not debug_mode else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() if debug_mode else logging.FileHandler(f'logs/training_{config.train_conf.run_name}.log'),
        ]
    )
    
    trainer = KG_LM_Trainer(
        config, 
        run_name=config.train_conf.run_name, 
        resume=config.train_conf.resume,
        enable_wandb=not debug_mode,
        save_checkpoints=not debug_mode,
    )
    
    try:
        trainer.train(time_budget_s=args.time_budget)
        trainer.close()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        
        if wandb.run:
            wandb.finish()
        trainer.close()
    except Exception as e:
        logging.error(f"Training failed with an unexpected error: {e}", exc_info=True)
        
        if wandb.run:
            wandb.finish()
        trainer.close()
        raise

if __name__ == "__main__":
    main()