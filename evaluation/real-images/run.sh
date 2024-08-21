#! /bin/bash
# "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf" #"llava-hf/llava-v1.6-34b-hf"  
# for model in "llava-hf/llava-v1.6-mistral-7b-hf" "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf" 
# for model in "BAAI/Bunny-v1_0-4B" "BAAI/Bunny-v1_0-3B" "BAAI/Bunny-v1_1-4B" "BAAI/Bunny-Llama-3-8B-V" "llava-hf/llava-v1.6-mistral-7b-hf" "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf" 
# for model in  "BAAI/Bunny-Llama-3-8B-V" "BAAI/Bunny-v1_0-3B"
# # for model in "adept/fuyu-8B"
# do
#     for type in "color" #"labelled" "labelled_id"
#     do
#         # python test-mcq-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-3-shapes-${type}-200.jsonl
#         # python test-tf-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-3-shapes-${type}-200.jsonl
#         python test-mcq-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-5-shapes-${type}-200.jsonl
#         # python test-tf-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-5-shapes-${type}-200.jsonl
#     done
# done

for model in  "llava-hf/llava-v1.6-mistral-7b-hf" "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf"  "BAAI/Bunny-v1_0-4B" "BAAI/Bunny-v1_0-3B" "BAAI/Bunny-v1_1-4B" "BAAI/Bunny-Llama-3-8B-V"  "Salesforce/instructblip-flan-t5-xl" "Salesforce/instructblip-vicuna-7b" "adept/fuyu-8B"
do
    python test-tf.py --model $model
done

exit
