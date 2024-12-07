import os
import json
import itertools
import base64
from time import sleep
import re
# from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import random
import argparse
import numpy as np

from prompt_formatters import depth_mcq_labelled_prompt, depth_mcq_color_prompt, height_mcq_color_prompt, height_mcq_labelled_prompt
from utils import get_prefix_suffix, get_ordinal, extract_shapes_from_ground_truth, extract_shape_counts

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str, default="../standard_data/depth_synthetic_2D/images-5-shapes-color-200.jsonl")
parser.add_argument("--img_dir", type=str, default="../../data")
parser.add_argument("--output_dir", type=str, default="../outputs")
parser.add_argument("--model", type=str, default="gpt4o")
parser.add_argument("--fewshot", type=str, default="True")
parser.add_argument("--self", type=str, default="False")
parser.add_argument("--k", type=str, default= 2)
parser.add_argument("--randomk", type=str, default="False")
parser.add_argument("--same_count", type=str, default="True")


args = parser.parse_args()

device = 'cuda' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)



if "gemini" in args.model:
    from utils import GeminiModels, gemini_config_vision

    LMMmodels = GeminiModels(gemini_config_vision)

elif "claude" in args.model:
    from utils import CLAUDEModels, claude_config

    LMMmodels = CLAUDEModels(claude_config)

elif "gptv" in args.model:
    from utils import OpenAIModelsAzure, openai_trnllm_gpt4turbov_config

    LMMmodels = OpenAIModelsAzure(openai_trnllm_gpt4turbov_config)
elif "gpt4o" in args.model:
    from utils import OpenAIModelsAzure, openai_config_gpt4o_azure

    LMMmodels = OpenAIModelsAzure(openai_config_gpt4o_azure)

eval_output_path = os.path.join(args.output_dir, f"{args.model.replace('/','-')}-eval.jsonl")

# prefix, suffix = get_prefix_suffix(args.model)


def evaluate_model(model, prompts_file, img_dir, file):
    accuracies = []

    for run in range(5):
        total = 0
        correct = 0

        with open(prompts_file, 'r') as json_file:
            json_list = list(json_file)
        count = 0
        for json_str in json_list:
            if count>=50:
                break
            count += 1
            prompt_data = json.loads(json_str)

            # prompt_text = prompt_data["prompt"]
            answer_set = prompt_data["options"]
            answer = prompt_data["ground_truth"]
            img_path = os.path.join(img_dir, prompt_data["img_path"])
            question_items = prompt_data["question_items"]
            
            if "depth" in prompts_file:
                if "color" in prompts_file:
                    prompt_text = depth_mcq_color_prompt(question_items, answer_set)
                else:
                    prompt_text = depth_mcq_labelled_prompt(question_items, answer_set)

            else:
                prompt_text = prompt_data["prompt_text"]
                num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
                num_stacks = int(num_stacks)
                if "color" in prompts_file:
                    prompt_text = height_mcq_color_prompt(prompt_text, answer_set, question_items, num_stacks)
                else:
                    prompt_text = height_mcq_labelled_prompt(prompt_text, answer_set, question_items, num_stacks)
            
            # prompt_text = prefix + prompt_text + suffix
            
            result = model([img_path], [prompt_text], answer_set, answer, file)

            if (result == 1):
                correct += 1
            total += 1
        print(correct, total)
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Run {run + 1} accuracy: {accuracy}")


    mean_accuracy = np.mean(accuracies)
    std = np.std(accuracies)
    # print("Accuracy: ", correct/total)
    print("Total: ", total)
    # print("Correct: ", correct)
    # print(args, "MCQ")
    with open(os.path.join(eval_output_path), "a") as file:
        result = {
            'path': args.prompts_file,
            'type': 'MCQ',
            'total': total,
            'correct': correct,
            'accuracies': accuracies,
            'mean_accuracy': mean_accuracy,
            'std': std
            # 'accuracy': correct/total
        }
        file.write(json.dumps(result) + '\n')

def evaluate_model_fewshot_self(model, prompts_file, img_dir, file):

    total = 0
    correct = 0

    with open(prompts_file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
            
        prompt_data = json.loads(json_str)

        # prompt_text = prompt_data["prompt"]
        answer_set = prompt_data["options"]
        answer = prompt_data["ground_truth"]
        img_path = os.path.join(img_dir, prompt_data["img_path"])
        question_items = prompt_data["question_items"]
        
        if "depth" in prompts_file:
            if "color" in prompts_file:
                prompt_text = depth_mcq_color_prompt(question_items, answer_set)
            else:
                prompt_text = depth_mcq_labelled_prompt(question_items, answer_set)
        else:
            prompt_text = prompt_data["prompt_text"]
            num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
            num_stacks = int(num_stacks)
            if "color" in prompts_file:
                prompt_text = height_mcq_color_prompt(prompt_text, answer_set, question_items, num_stacks)
            else:
                prompt_text = height_mcq_labelled_prompt(prompt_text, answer_set, question_items, num_stacks)
        
        prompt_text_prefix_example = "Below is an example and solution of the task you will perform on the first image.\nExample:\n" + prompt_text + "\n" + answer     
        prompt_text = "\n\nNow solve the same task for the below example for the second image:\n" + prompt_text
        print("1)", prompt_text_prefix_example)
        print("2)", prompt_text)

        result = model([img_path, img_path], [prompt_text_prefix_example, prompt_text], answer_set, answer, file)

        if (result == 1):
            correct += 1

        total += 1
        print(correct, total)

    print("Accuracy: ", correct/total)
    print("Total: ", total)
    print("Correct: ", correct)
    print(args, "MCQ")
    with open(os.path.join(eval_output_path), "a") as file:
        result = {
            'path': args.prompts_file,
            'type': 'MCQ',
            'total': total,
            'correct': correct,
            'accuracy': correct/total
        }
        file.write(json.dumps(result) + '\n')


def evaluate_model_fewshot(model, prompts_file, img_dir, file, k):
    accuracies = []

    for run in range(5):
        total = 0
        correct = 0

        with open(prompts_file, 'r') as json_file:
            json_list = [json.loads(json_str) for json_str in json_file]

        for i, prompt_data in enumerate(json_list):
            if i >= 50:
                break
            # Get current example details
            answer_set = prompt_data["options"]
            answer = prompt_data["ground_truth"]
            img_path = os.path.join(img_dir, prompt_data["img_path"])
            question_items = prompt_data["question_items"]

            # Generate the prompt text for the current example
            if "depth" in prompts_file:
                if "color" in prompts_file:
                    prompt_text = depth_mcq_color_prompt(question_items, answer_set)
                else:
                    prompt_text = depth_mcq_labelled_prompt(question_items, answer_set)
            else:
                prompt_text = prompt_data["prompt_text"]
                num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
                num_stacks = int(num_stacks)
                if "color" in prompts_file:
                    prompt_text = height_mcq_color_prompt(prompt_text, answer_set, question_items, num_stacks)
                else:
                    prompt_text = height_mcq_labelled_prompt(prompt_text, answer_set, question_items, num_stacks)

            # Randomly select `k` examples excluding the current one
            examples = random.sample(
                [example for j, example in enumerate(json_list) if j != i],  # Exclude current example
                min(k, len(json_list) - 1)  # Ensure we don't exceed available examples
            )

            # Prepare prompts and images for k-shot learning
            k_shot_prompts = []
            k_shot_images = []

            for idx, example in enumerate(examples):
                example_img_path = os.path.join(img_dir, example["img_path"])
                example_answer_set = example["options"]
                example_answer = example["ground_truth"]
                example_question_items = example["question_items"]

                if "depth" in prompts_file:
                    if "color" in prompts_file:
                        example_prompt_text = depth_mcq_color_prompt(example_question_items, example_answer_set)
                    else:
                        example_prompt_text = depth_mcq_labelled_prompt(example_question_items, example_answer_set)
                else:
                    num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
                    num_stacks = int(num_stacks)
                    if "color" in prompts_file:
                        example_prompt_text = height_mcq_color_prompt(example["prompt_text"], example_answer_set, example_question_items, num_stacks)
                    else:
                        example_prompt_text = height_mcq_labelled_prompt(example["prompt_text"], example_answer_set, example_question_items, num_stacks)

                # Add prompt_text_prefix_example for this example
                example_prompt = (
                    f"Below is an example and solution of the task you will perform on the {get_ordinal(idx + 1)} image.\n"
                    f"Example:\n{example_prompt_text}\nSolution: {example_answer}"
                )

                # Append to the lists
                k_shot_prompts.append(example_prompt)
                k_shot_images.append(example_img_path)

            # Add the target question with a specific reference
            target_prompt = (
                f"Now solve the same task for the below example:\n{prompt_text}"
            )
            k_shot_prompts.append(target_prompt)
            k_shot_images.append(img_path)

            print("Prompts and images sent to the model:")
            for idx, (prompt, image) in enumerate(zip(k_shot_prompts, k_shot_images)):
                print(f"Prompt {idx + 1}: {prompt}\nImage: {image}\n")

            # Call the model with paired prompts and images
            result = model(k_shot_images, k_shot_prompts, answer_set, answer, file)

            if result == 1:
                correct += 1

            total += 1
        print(correct, total)
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Run {run + 1} accuracy: {accuracy}")

    mean_accuracy = np.mean(accuracies)
    std = np.std(accuracies)

    # Output final accuracy
    # print("Accuracy: ", correct / total)
    print("Total: ", total)
    # print("Correct: ", correct)

    with open(os.path.join(eval_output_path), "a") as file:
        result = {
            'path': prompts_file,
            'type': 'MCQ',
            'total': total,
            'correct': correct,
            'accuracies': accuracies,
            'mean_accuracy': mean_accuracy,
            'std': std,
            # 'accuracy': correct / total,
            'k': k
        }
        file.write(json.dumps(result) + '\n')

def evaluate_model_fewshot_same_shapes(model, prompts_file, img_dir, file, k):
    accuracies = []

    for run in range(5):
        total = 0
        correct = 0

        # Load all examples from the prompts file
        with open(prompts_file, 'r') as json_file:
            json_list = [json.loads(json_str) for json_str in json_file]

        for i, prompt_data in enumerate(json_list):
            if i >= 50:
                break
            # Get current example details
            answer_set = prompt_data["options"]
            answer = prompt_data["ground_truth"]
            img_path = os.path.join(img_dir, prompt_data["img_path"])
            question_items = prompt_data["question_items"]

            # Filter examples with matching shapes
            if args.same_count == "True":
                # Extract shapes from the test example's ground truth
                test_shape_counts = extract_shape_counts(answer)

                matching_examples = [
                    example for j, example in enumerate(json_list)
                    if j != i and extract_shape_counts(example["ground_truth"]) == test_shape_counts
                ]
            else:
                # Extract shapes from the test example's ground truth
                test_shapes = extract_shapes_from_ground_truth(answer)

                matching_examples = [
                example for j, example in enumerate(json_list)
                if j != i and extract_shapes_from_ground_truth(example["ground_truth"]) == test_shapes
            ]

            # Randomly select k examples from the matching set
            selected_examples = random.sample(matching_examples, min(k, len(matching_examples)))

            # Generate the prompt text for the current example
            if "depth" in prompts_file:
                if "color" in prompts_file:
                    prompt_text = depth_mcq_color_prompt(question_items, answer_set)
                else:
                    prompt_text = depth_mcq_labelled_prompt(question_items, answer_set)
            else:
                prompt_text = prompt_data["prompt_text"]
                num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
                num_stacks = int(num_stacks)
                if "color" in prompts_file:
                    prompt_text = height_mcq_color_prompt(prompt_text, answer_set, question_items, num_stacks)
                else:
                    prompt_text = height_mcq_labelled_prompt(prompt_text, answer_set, question_items, num_stacks)

            # Prepare prompts and images for k-shot learning
            k_shot_prompts = []
            k_shot_images = []

            for idx, example in enumerate(selected_examples):
                example_img_path = os.path.join(img_dir, example["img_path"])
                example_answer_set = example["options"]
                example_answer = example["ground_truth"]
                example_question_items = example["question_items"]

                if "depth" in prompts_file:
                    if "color" in prompts_file:
                        example_prompt_text = depth_mcq_color_prompt(example_question_items, example_answer_set)
                    else:
                        example_prompt_text = depth_mcq_labelled_prompt(example_question_items, example_answer_set)
                else:
                    num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
                    num_stacks = int(num_stacks)
                    if "color" in prompts_file:
                        example_prompt_text = height_mcq_color_prompt(example["prompt_text"], example_answer_set, example_question_items, num_stacks)
                    else:
                        example_prompt_text = height_mcq_labelled_prompt(example["prompt_text"], example_answer_set, example_question_items, num_stacks)

                # Add prompt_text_prefix_example for this example
                example_prompt = (
                    f"Below is an example and solution of the task you will perform on the {get_ordinal(idx + 1)} image.\n"
                    f"Example:\n{example_prompt_text}\nSolution: {example_answer}"
                )


                # Append to the lists
                k_shot_prompts.append(example_prompt)
                k_shot_images.append(example_img_path)

            # Add the target question with a specific reference
            target_prompt = (
                f"Now solve the same task for the below example:\n{prompt_text}"
            )
            k_shot_prompts.append(target_prompt)
            k_shot_images.append(img_path)

            print("Prompts and images sent to the model:")
            for idx, (prompt, image) in enumerate(zip(k_shot_prompts, k_shot_images)):
                print(f"Prompt {idx + 1}: {prompt}\nImage: {image}\n")

            # Call the model with paired prompts and images
            result = model(k_shot_images, k_shot_prompts, answer_set, answer, file)

            if result == 1:
                correct += 1

            total += 1

        print(correct, total)
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Run {run + 1} accuracy: {accuracy}")

    mean_accuracy = np.mean(accuracies)
    std = np.std(accuracies)


    # Output final accuracy
    # print("Accuracy: ", correct / total)
    print("Total: ", total)
    # print("Correct: ", correct)

    with open(os.path.join(eval_output_path), "a") as file:
        result = {
            'path': prompts_file,
            'type': 'MCQ',
            'total': total,
            'correct': correct,
            'accuracies': accuracies,
            'mean_accuracy': mean_accuracy,
            'std': std,
            # 'accuracy': correct / total,
            'k': k,
            'same_shape': "True",
            'same_count': args.same_count
        }
        file.write(json.dumps(result) + '\n')


def model_gptv(img_path, prompt_text, answer_set, answer, file):
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
    request_data['messages'][1]["content"] = [
            {"image": encoded_image},
            prompt_text
    ]
    try:
        response = llm_client.send_request('dev-gpt-4v-chat-completions', request_data)
        predicted_answer = response["choices"][0]["message"]["content"]
        predicted_answer = predicted_answer.replace("'","") # remove single quotes, not sure why it is appearing
        predicted_answer = predicted_answer.replace(' ','') # remove spaces         
    except Exception as e:
        print(e)
        sleep(10)
        predicted_answer = "Error"
    
    answer = answer.replace(' ','') # remove spaces
    judgement = int(predicted_answer == answer)
    result = {
        'img_path': img_path,
        'prompt_text': prompt_text,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'judgement': judgement
    }
    file.write(json.dumps(result) + '\n')
    
    return judgement

#### Gemini model ###

def model_gemini(img_path, prompt_text, answer_set, answer, file):
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')

    response = LMMmodels.generate(prompt_text, [encoded_image]) 
    
    predicted_answer = response[0].strip()
    predicted_answer = predicted_answer.replace("'","") # remove single quotes, not sure why it is appearing
    predicted_answer = predicted_answer.replace(' ','') # remove spaces         
   
    answer = answer.replace(' ','') # remove spaces
    judgement = int(predicted_answer == answer)
    result = {
        'img_path': img_path,
        'prompt_text': prompt_text,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'response': response[0],
        'judgement': judgement
    }
    file.write(json.dumps(result) + '\n')
    
    return judgement


def model_gpt4o(img_paths, prompt_texts, answer_set, answer, file):
    # Generate the response using the OpenAI API model
    response = LMMmodels.generate(prompt_texts, img_paths)
    print('output:',response)
    # Process the predicted answer to match format
    predicted_answer = response.strip()
    predicted_answer = predicted_answer.replace("'", "")  # Remove single quotes
    predicted_answer = predicted_answer.replace(' ', '')  # Remove spaces
    
    # Compare prediction with the ground truth answer
    answer = answer.replace(' ','')  # Remove spaces
    judgement = int(predicted_answer == answer)
    # Create the result dictionary
    result = {
        'img_path': img_paths,
        'prompt_text': prompt_texts,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'response': response,
        'judgement': judgement
    }

    # Write the result to the output file
    file.write(json.dumps(result) + '\n')
    file.flush()  # Ensure the write is flushed to the file
    
    return judgement

####################






#######################

if __name__ == "__main__":

    PATH_TO_FOLDER = args.prompts_file

    IMG_DIR = args.img_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    last_dir_name = last_dir_name.split(".")[0]
    os.makedirs(f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/mcq.jsonl"
    if args.fewshot == 'True' and args.self == 'False':
            if args.randomk == 'True':
                output_path = f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/mcq-fewshot-{args.k}.jsonl"
            else:
                output_path = f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/mcq-fewshot-{args.k}-samecolor.jsonl"
    
    print(output_path)
    if 'gptv' in args.model:
        model = model_gemini
    elif 'gemini' in args.model:
        model = model_gemini
    elif 'claude' in args.model:
        model = model_gemini
    elif 'gpt4o' in args.model:
        if 'False' in args.fewshot:
            model = model_gpt4o
        else:
            print('fewshot')
            model = model_gpt4o     
    with open(output_path, "w") as file:
        if 'False' in args.fewshot:
            evaluate_model(model, PATH_TO_FOLDER, IMG_DIR, file)
        else:
            if 'False' in args.self:
                if 'False' in args.randomk:
                    evaluate_model_fewshot_same_shapes(model,PATH_TO_FOLDER,IMG_DIR,file,args.k)
                else:
                    evaluate_model_fewshot(model,PATH_TO_FOLDER,IMG_DIR,file,args.k)

            else:
                evaluate_model_fewshot_self(model, PATH_TO_FOLDER, IMG_DIR, file)
