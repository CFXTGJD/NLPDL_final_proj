from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import random
import json
from datasets import load_dataset_builder
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name, sep_token=" "):
    # '''
	# dataset_name: str, the name of the dataset
	# sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	# '''

    def merge_options(options, dataset_name):
        if dataset_name == "med_qa":
            return " ".join([f"{option['key']}:{option['value']}" for i, option in enumerate(options)])
        elif dataset_name == "jec_qa_KD" or dataset_name == "jec_qa_CA":
            return ' '.join(f'{key}:{value}' for key, value in options.items())
        
    def merge_multi_answers(answers: list):
        answer_str = ""
        for answer in answers:
            answer_str += answer + ", "
        return answer_str[:-2]

    dataset = None
    if isinstance(dataset_name, str):
        if dataset_name == "med_qa":
            train_data_path = "/ceph/home/liuxingyu/NLP/final/datasets/med_qa/train/med_qa_train.json"
            with open(train_data_path, 'r', encoding='utf-8') as file:
                train_data = json.load(file)
            test_data_path = "/ceph/home/liuxingyu/NLP/final/datasets/med_qa/test/med_qa_test.json"
            with open(test_data_path, 'r', encoding='utf-8') as file:
                test_data = json.load(file)
            valid_data_path = "/ceph/home/liuxingyu/NLP/final/datasets/med_qa/validation/med_qa_val.json"
            with open(valid_data_path, 'r', encoding='utf-8') as file:
                valid_data = json.load(file)
            
            #train
            train_question_list = train_data["question"]
            train_options_list = train_data["options"]
            train_answer_idx_list = train_data["answer_idx"]
            train_answer_list = train_data["answer"]
            train_options_list = [merge_options(options, dataset_name) for options in train_options_list]

            train_input_list = [f"{question}{sep_token}{options}" for question, options in zip(train_question_list, train_options_list)]
            train_output_list = [f"{answer_idx}:{answer}" for answer_idx, answer in zip(train_answer_idx_list, train_answer_list)]

            train_tuple_list = list(zip(train_input_list, train_output_list))

            #test
            test_question_list = test_data["question"]
            test_options_list = test_data["options"]
            test_answer_idx_list = test_data["answer_idx"]
            test_answer_list = test_data["answer"]
            test_options_list = [merge_options(options, dataset_name) for options in test_options_list]

            test_input_list = [f"{question}{sep_token}{options}" for question, options in zip(test_question_list, test_options_list)]
            test_output_list = [f"{answer_idx}:{answer}" for answer_idx, answer in zip(test_answer_idx_list, test_answer_list)]

            test_tuple_list = list(zip(test_input_list, test_output_list))

            #valid
            valid_question_list = valid_data["question"]
            valid_options_list = valid_data["options"]
            valid_answer_idx_list = valid_data["answer_idx"]
            valid_answer_list = valid_data["answer"]
            valid_options_list = [merge_options(options, dataset_name) for options in valid_options_list]

            valid_input_list = [f"{question}{sep_token}{options}" for question, options in zip(valid_question_list, valid_options_list)]
            valid_output_list = [f"{answer_idx}:{answer}" for answer_idx, answer in zip(valid_answer_idx_list, valid_answer_list)]

            valid_tuple_list = list(zip(valid_input_list, valid_output_list))

            train_dataset = Dataset.from_dict({"tuples": train_tuple_list})
            test_dataset = Dataset.from_dict({"tuples": test_tuple_list})
            valid_dataset = Dataset.from_dict({"tuples": valid_tuple_list})
            
            dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": valid_dataset})

        elif dataset_name == "jec_qa_KD" or dataset_name == "jec_qa_CA":
            if dataset_name == "jec_qa_KD":
                data_path = "/ceph/home/liuxingyu/NLP/final/datasets/jec_qa_KD/KD_train.json"
            elif dataset_name == "jec_qa_CA":
                data_path = "/ceph/home/liuxingyu/NLP/final/datasets/jec_qa_CA/CA_train.json"
            data = []
            with open(data_path, 'r') as file:
                for line in file:
                    json_object = json.loads(line)
                    data.append(json_object)
            train_ratio = 0.8
            val_ratio = 0.15
            test_ratio = 0.05
            train_data, temp_data = train_test_split(data, test_size=1 - train_ratio, random_state=42)
            validation_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

            #train
            train_statement_list = [item['statement'] for item in train_data]
            train_options_list = [merge_options(item['option_list'], dataset_name) for item in train_data]
            train_answer_list = [merge_multi_answers(item['answer']) for item in train_data]

            train_input_list = [statement + sep_token + options for statement, options in zip(train_statement_list, train_options_list)]
            train_output_list = train_answer_list

            train_tuple_list = list(zip(train_input_list, train_output_list))

            #validation
            validation_statement_list = [item['statement'] for item in validation_data]
            validation_options_list = [merge_options(item['option_list'], dataset_name) for item in validation_data]
            validation_answer_list = [merge_multi_answers(item['answer']) for item in validation_data]

            validation_input_list = [statement + sep_token + options for statement, options in zip(validation_statement_list, validation_options_list)]
            validation_output_list = validation_answer_list

            validation_tuple_list = list(zip(validation_input_list, validation_output_list))

            #test
            test_statement_list = [item['statement'] for item in test_data]
            test_options_list = [merge_options(item['option_list'], dataset_name) for item in test_data]
            test_answer_list = [merge_multi_answers(item['answer']) for item in test_data]

            test_input_list = [statement + sep_token + options for statement, options in zip(test_statement_list, test_options_list)]
            test_output_list = test_answer_list

            test_tuple_list = list(zip(test_input_list, test_output_list))

            train_dataset = Dataset.from_dict({"tuples": train_tuple_list})
            validation_dataset = Dataset.from_dict({"tuples": validation_tuple_list})
            test_dataset = Dataset.from_dict({"tuples": test_tuple_list})

            dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

        return dataset

