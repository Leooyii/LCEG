import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["qasper", "trec", "triviaqa", "samsum", "lsht"] or "trec" in dataset:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer_by_length(dataset, predictions, answers, lengths, all_classes):
    total_score = 0.
    scores = { "0-4k": [], "4-8k": [], "8-16k": [], "16-32k":[], "4k+":[], "#0-4k": 0, "#4-8k": 0, "#8-16k": 0, "#16-32k": 0, "#4k+":0}
    detail_scores = {"length":[], "score":[]}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        post_process_list = [
                "narrativeqa", "qasper", "multifieldqa_en",
                "hotpotqa", "2wikimqa", "musique",
                "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en"
                ]
        if dataset in post_process_list:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        if length > 4000:
            scores["4k+"].append(score)
            scores["#4k+"] += 1
        if length < 4000:
            scores["0-4k"].append(score)
            scores["#0-4k"] += 1
        elif length < 8000:
            scores["4-8k"].append(score)
            scores["#4-8k"] += 1
        elif length < 16000:
            scores["8-16k"].append(score)
            scores["#8-16k"] += 1
        else:
            scores["16-32k"].append(score)
            scores["#16-32k"] += 1
        detail_scores["length"].append(length)
        detail_scores["score"].append(score)
    for key in ["0-4k", "4-8k", "8-16k", "16-32k", "4k+"]:
        scores[key] = round(100 * np.mean(scores[key]), 2)
    scores["total_score"] = round(100 * total_score / len(predictions), 2)
    return scores, detail_scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        post_process_list = [
                "narrativeqa", "qasper", "multifieldqa_en",
                "hotpotqa", "2wikimqa", "musique",
                "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en"
                ]
        if dataset in post_process_list:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    total_scores = dict()
    detail_scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        score, detail_score = scorer_by_length(dataset, predictions, answers, lengths, all_classes)
        scores[dataset] = score
        total_scores[dataset] = score["total_score"]
        detail_scores[dataset] = detail_score
    if args.e:
        out_path_result = f"pred/{args.model}/result.json"
        out_path_result_by_length = f"pred_e/{args.model}/result_by_length.json"
        out_path_result_by_details = f"pred_e/{args.model}/result_by_details.json"
    else:
        out_path_result = f"pred/{args.model}/result.json"
        out_path_result_by_length = f"pred/{args.model}/result_by_length.json"
        out_path_result_by_details = f"pred/{args.model}/result_by_details.json"

    with open(out_path_result, "w") as f:
        json.dump(total_scores, f, ensure_ascii=False, indent=4)
    with open(out_path_result_by_length, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    with open(out_path_result_by_details, "w") as f:
        json.dump(detail_scores, f, ensure_ascii=False, indent=4)
