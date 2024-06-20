import os
import json
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch


def get_rank(entry):
	return entry[1]["rank"]

def main():
	token_f = open("token.txt", "r")
	token = token_f.read()
	login(token.strip())

	device = "cuda"

	model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

	dir_path = './data_construction/'
	dir_path_out = './data_generated/'

	discussions_path = f'{dir_path}discussions_final/'
	files_disc = os.listdir(discussions_path)
	dev_disc = [files_disc.pop(3),files_disc.pop(4), files_disc.pop(6)]
	eval_disc = files_disc
	
	json_path = f'{dir_path}summaries_json/'
	files_json = os.listdir(json_path)
	dev_json = [files_json.pop(3), files_json.pop(4), files_json.pop(6)]
	eval_json = files_json

	dev_json_data = []
	for f in dev_json:
		jsonf = open(json_path+f, "r")
		dev_json_data.append(json.load(jsonf))
	
	dev_disc_data = []
	for f in dev_disc:
		discf = open(discussions_path+f, "r")
		dev_disc_data.append(discf.read())

	sorted_summaries = []
	sorted_explanations = []
	for x in dev_json_data:
		# Create a list of tuples containing the source name and its corresponding data
		entries = [(key, value) for key, value in x.items()]

		# Sort the list based on the rank
		sorted_entries = sorted(entries, key=get_rank)

		# Collect summaries and explanations for the current document
		current_summaries = []
		current_explanation = []
		for entry in sorted_entries:
			current_summaries.append(entry[1])
			explanation = ""
			for key, value in entry[1]["quality"].items():
				explanation += f"{key}: {', '.join([str(value[0]), value[1]])}\n\n"
			current_explanation.append(explanation.strip())

		sorted_summaries.append(current_summaries)
		sorted_explanations.append(current_explanation)
	
	prompt_original = "[START OF DOCUMENT]\n{0}[END OF DOCUMENT]\n\nFirstly, report the most important findings in this study in one sentence. Afterwards, review the text and decide whether the authors explicitly acknowledge any limitations of the paper. If limitations are mentioned, summarize them briefly in at most 300 words. Do not use bullet points. Only describe the limitations that are explicitly mentioned by the author. If the author did not explicitly acknowledge limitations, simply reply with ’No limitations are provided by the authors'."

	prompt_improved = "[START OF DOCUMENT]\n{0}[END OF DOCUMENT]\n\nFirstly, report the most important finding in this study in one sentence. Make sure to keep this very brief, and only include the most important information. Afterwards, review the text and decide whether the authors explicitly acknowledge any limitations of the paper. Only if limitations are mentioned explicitly, summarize them briefly in at most 300 words. Do not use bullet points. Do not derive potential limitations from the description or the results mentioned in the text. Only describe the limitations that are explicitly mentioned by the authors. If the authors did not explicitly acknowledge limitations in this text, simply reply with 'No limitations are provided by the authors'. If you are unsure whether an aspect of the text is mentioned as a limitation or not, explain this instead."

	prompt_article_top = ["I will provide you with a few examples of discussion sections of scientific documents. For each of these examples, a prompt to create a brief summary is included. After this prompt, there will be an example of a high quality output for this prompt. After inspecting these examples, reply with 'example received'.\n\nAfter these examples, I will provide you with a new text, followed by the same prompt. You will then generate an output to this prompt yourself, inspired by the examples I have provided. If you understand the task, please reply with 'understood'.", "[EXAMPLE]\n"+prompt_improved+"\n\n{1}"]

	prompt_article_all = ["I will provide you with a few examples of discussion sections of scientific documents. For each of these examples, a prompt to create a brief summary is included. After this prompt, there will be 3 example outputs. These output are all ranked from 1 to 3. The highest quality summary will receive the rank of 1, while the lowest quality summary will receive the rank of 3. After inspecting these examples, reply with 'example received'.\n\nAfter these examples, I will provide you with a new text, followed by the same prompt. You will then generate an output to this prompt yourself, inspired by the examples I have provided. If you understand the task, please reply with 'understood'.", "[EXAMPLE]\n"+prompt_improved+"\n\n{1}\n[RANK: {2}]\n\n{3}\n[RANK: {4}]\n\n{5}\n[RANK: {6}]"]

	prompt_article_all_explanation = ["I will provide you with a few examples of discussion sections of scientific documents. For each of these examples, a prompt to create a brief summary is included. After this prompt, there will be 3 example outputs. These output are all ranked from 1 to 3. The highest quality summary will receive the rank of 1, while the lowest quality summary will receive the rank of 3. Additionally, each summary contains an explanation on its quality. Each summary is rated on the basis of factual consistency, coverage and coherence.\n\nFor factual consistency, a score of 1 is assigned when all information is in line with the original text. A score of 2 is assigned, for example when the information is correct but contains little to no detail. A score of 3 is assigned when the summary contains information that is not in line with the original text.\n\nFor coverage, a score of 1 is assigned when the summary covers all limitations mentioned in the original text. A score of 2 is assigned when all limitations are mentioned, but not explained. A score of 3 is assigned when the summary fails to mention all limitations.\n\nFor coherence, a score of 1 is assigned when the summary is coherent and does not contain hallucinations. A score of 2 is assigned when the summary does not follow the instructions in the prompt perfectly, but is still coherent and does not include hallucinations. A score of 3 is assigned when the summary contains hallucinations or is incomprehensible, for example when abbreviations are used but never explained.\n\n After inspecting these examples, reply with 'example received'.\n\nAfter these examples, I will provide you with a new text, followed by the same prompt. You will then generate an output to this prompt yourself, inspired by the examples I have provided. If you understand the task, please reply with 'understood'.", "[EXAMPLE]\n"+prompt_improved+"\n\n{1}\n[RANK: {2}]\n[EXPLANATION: {3}]\n\n{4}\n[RANK: {5}]\n[EXPLANATION: {6}]\n\n{7}\n[RANK: {8}]\n[EXPLANATION: {9}]"]

	for f in eval_disc:
		discf = open(discussions_path+f, "r")
		discussion = discf.read()
		discf.close()
		id = f[:-4]
	
		prompts = {
			"zeroshot_original": [
				{
					"role": "user",
					"content": prompt_original.format(discussion)
				}
			],
			"zeroshot_improved": [
				{
					"role": "user",
					"content": prompt_improved.format(discussion)
				}
			],
			"article_top": [
				{
					"role": "user",
					"content": prompt_article_top[0]

				},
				{
					"role": "assistant",
					"content": "Understood."
				},
				{
					"role": "user",
					"content": prompt_article_top[1].format(dev_disc_data[0], sorted_summaries[0][0]["summ"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_top[1].format(dev_disc_data[1], sorted_summaries[1][0]["summ"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_top[1].format(dev_disc_data[2], sorted_summaries[2][0]["summ"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_improved.format(discussion)
				}
			],
			"article_all": [
				{
					"role": "user",
					"content": prompt_article_all[0]
				},
				{
					"role": "assistant",
					"content": "Understood."
				},
				{
					"role": "user",
					"content": prompt_article_all[1].format(dev_disc_data[0], sorted_summaries[0][0]["summ"], sorted_summaries[0][0]["rank"],  sorted_summaries[0][1]["summ"], sorted_summaries[0][1]["rank"],  sorted_summaries[0][2]["summ"], sorted_summaries[0][2]["rank"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_all[1].format(dev_disc_data[1], sorted_summaries[1][0]["summ"], sorted_summaries[1][0]["rank"],  sorted_summaries[1][1]["summ"], sorted_summaries[1][1]["rank"],  sorted_summaries[1][2]["summ"], sorted_summaries[1][2]["rank"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_all[1].format(dev_disc_data[2], sorted_summaries[2][0]["summ"], sorted_summaries[2][0]["rank"],  sorted_summaries[2][1]["summ"], sorted_summaries[2][1]["rank"],  sorted_summaries[2][2]["summ"], sorted_summaries[2][2]["rank"])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_improved.format(discussion)
				}
			],
			"article_all_explanation": [
				{
					"role": "user",
					"content": prompt_article_all_explanation[0]
				},
				{
					"role": "assistant",
					"content": "Understood."
				},
				{
					"role": "user",
					"content": prompt_article_all_explanation[1].format(dev_disc_data[0], sorted_summaries[0][0]["summ"], sorted_summaries[0][0]["rank"], sorted_explanations[0][0], sorted_summaries[0][1]["summ"], sorted_summaries[0][1]["rank"],  sorted_explanations[0][1], sorted_summaries[0][2]["summ"], sorted_summaries[0][2]["rank"], sorted_explanations[0][2])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_all_explanation[1].format(dev_disc_data[1], sorted_summaries[1][0]["summ"], sorted_summaries[1][0]["rank"], sorted_explanations[1][0], sorted_summaries[1][1]["summ"], sorted_summaries[1][1]["rank"], sorted_explanations[1][1], sorted_summaries[1][2]["summ"], sorted_summaries[1][2]["rank"], sorted_explanations[1][2])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_article_all_explanation[1].format(dev_disc_data[2], sorted_summaries[2][0]["summ"], sorted_summaries[2][0]["rank"], sorted_explanations[2][0], sorted_summaries[2][1]["summ"], sorted_summaries[2][1]["rank"], sorted_explanations[2][1], sorted_summaries[2][2]["summ"], sorted_summaries[2][2]["rank"], sorted_explanations[2][2])
				},
				{
					"role": "assistant",
					"content": "Example received."
				},
				{
					"role": "user",
					"content": prompt_improved.format(discussion)
				}
			]
		}

		if len(discussion.split()) > 10000:
			print("TEST")
			prompts["zeroshot_original"] = [
				{
					"role": "user",
					"content": "[START OF PART 1]\n{0}[END OF PART 1]\n\For now, only reply with 'understood'".format(discussion.split()[:10000])
				},
				{
					"role": "assistant",
					"content": "Understood."
				},
				{
					"role": "user",
					"content": "[START OF PART 2]\n{0}[END OF PART 2]\n\nFirstly, report the most important findings in this study in one sentence. Afterwards, review the text and decide whether the authors explicitly acknowledge any limitations of the paper. If limitations are mentioned, summarize them briefly in at most 300 words. Do not use bullet points. Only describe the limitations that are explicitly mentioned by the author. If the author did not explicitly acknowledge limitations, simply reply with ’No limitations are provided by the authors'.".format(discussion.split()[10000:])
				}
			]
			prompts["zeroshot_improved"] = [
				{
					"role": "user",
					"content": "[START OF PART 1]\n{0}[END OF PART 1]\n\For now, only reply with 'understood'".format(discussion.split()[:10000])
				},
				{
					"role": "assistant",
					"content": "Understood."
				},
				{
					"role": "user",
					"content": "[START OF PART 2]\n{0}[END OF PART 2]\n\nFirstly, report the most important finding in this study in one sentence. Make sure to keep this very brief, and only include the most important information. Afterwards, review the text and decide whether the authors explicitly acknowledge any limitations of the paper. Only if limitations are mentioned explicitly, summarize them briefly in at most 300 words. Do not use bullet points. Do not derive potential limitations from the description or the results mentioned in the text. Only describe the limitations that are explicitly mentioned by the authors. If the authors did not explicitly acknowledge limitations in this text, simply reply with 'No limitations are provided by the authors'. If you are unsure whether an aspect of the text is mentioned as a limitation or not, explain this instead.".format(discussion.split()[10000:])
				}
			]

		list1 = os.listdir(dir_path_out+'article_all')
		list2 = os.listdir(dir_path_out+'article_all_explanation')
		list3 = os.listdir(dir_path_out+'article_top')
		list4 = os.listdir(dir_path_out+'zeroshot_improved')
		list5 = os.listdir(dir_path_out+'zeroshot_original')

		if not all(id+'.txt' in list for list in [list1, list2, list3, list4, list5]):
			for prompt_type in prompts:
				torch.cuda.empty_cache()
				messages = prompts[prompt_type]

				encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

				model_inputs = encodeds.to(device)
				model.to(device)

				generated_ids = model.generate(model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500, do_sample=True)

				# length of the input
				input_length = model_inputs.shape[1]

				# slice generated_ids tensor
				output_ids = generated_ids[:, input_length:]

				# decode output IDs
				decoded = tokenizer.batch_decode(output_ids)

				f = open(f"{dir_path_out}{prompt_type}/{id}.txt", "w")
				
				f.write(decoded[0].replace("</s>", ""))
				f.close()


if __name__=="__main__":
	main()

