#!/bin/bash
#SBATCH --job-name=eval                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=1-00:00:00                 # time limits
#SBATCH --output=out/eval_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:4                    # Generic resources, e.g., GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --chdir=.                       # start from current directory
#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="configs/base_config.yaml"
else
    export CONFIG_FILE=$1
fi

OUTPUT_FILE="${2:-eval/eval-$JOB_ID.json}"
BATCH_SIZE="${3:-16}"
MAX_SAMPLES="${4:-None}"
SPLIT="${5:-test}"

accelerate launch --config-file configs/accelerate_evalconfig.yaml evaluate_thinking.py --config $CONFIG_FILE --output_file $OUTPUT_FILE --batch_size $BATCH_SIZE --max_samples $MAX_SAMPLES --no_baseline --split $SPLIT
