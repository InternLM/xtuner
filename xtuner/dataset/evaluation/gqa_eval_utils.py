# Evaluation code for GQA.
# Computes a suite of metrics such as accuracy, consistency, plausibility and scores per question type and length.
# Visit https://gqadataset.org/ for all information about the dataset, including examples, visualizations, paper and slides.
#
#
# Metrics:
# - Accuracy: Standard accuracy, computed over the balanced version of the dataset, which is more robust against
#             cheating by making educated guesses. For each question-answer pair (q,a), we give 1 point if the
#             predicted answer p matches a and 0 otherwise, and average over all questions in the dataset.
#
# - Consistency: A metric for the level of model's consistency across different questions. For each question-answer
#                pair (q,a), we define a set Eq={q1, q2, ..., qn} of entailed questions, the answers to which can
#                be unambiguously inferred given (q,a).
#                Denote Q the set of all questions the model answered correctly. For each question q in Q, we
#                measure the model's accuracy over the entailed questions Eq to get the score sq and finally
#                average these results across all questions in Q.
#
# - Validity: Measures whether the model gives a "valid" answer - one that can theoretically be an answer
#             to the question (e.g. a color to a color question, yes/no to a binary question etc.).
#             We provide a set of valid answers to each questions over the final answer vocabulary, in
#             the choices file, and use it to compute average validity across the dataset.
#
# - Plausibility: Measures whether the model answers are plausible, e.g. one that make sense in the real world,
#                 e.g. not answering "purple" to a question about apple color (unless it's really purple).
#                 We provide a set of all plausible answers to each questions, computed by looking at all
#                 attributes and relations hold for various objects throughout the whole dataset scene graphs,
#                 and use it to compute average model plausibility across the data.
#
# - Grounding: Only for attention models. Measures whether the model looks at the relevant regions in the
#              image when answering a question. Each question in the dataset is annotated with the visual regions
#              they refer to, which are then used to compute the level to which the model has a correct visual attention,
#              which will allow to identify whether it really answers based on the image of by language-based guesses.
#              Supports both spatial features and object-based features.
#
# - Distribution: Measures the overall match between the true answer distribution for different questions,
#                 vs the overall distribution predicted by the model through its answers for all the data.
#                 We use chi-square statistic to measure the degree of similarity between the distributions,
#                 giving indication to the level of overall world-knowledge of the model
#
# - Accuracy per type: accuracy per question structural types (logic, compare, choose), and semantic type
#                      (questions about attributes, relations, categories, objects or the whole scene).
#
# - Accuracy for length: accuracy as a function of the question length, in terms of (1) words number, and semantic
#                        complexity - number of reasoning steps.
#
# We may support additional metrics (e.g. coverage) in the future.
#
#
# Files format:
# - predictions file format: JSON array: [{"questionId": str, "prediction": str}]
# - attentions file format: JSON array:
#   Spatial attention: [{"questionId": str, "attention": [mapSize x mapSize: float] }].
#   Object-based attention:[{"questionId": str, "attention": [[x0, y0, x1, y1, float] x #regions] }]. 0 < x,y < 1.
# - questions and choices files are provided as part of the dataset.
#   see https://gqadataset.org/download.html for information about their format.
#
#
# If you have any questions or comments, please feel free to send an email,
# at dorarad@cs.stanford.edu. We hope you'll enjoy using the GQA dataset! :)
#
#
# import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import os.path
import glob
import json
from mmengine.logging import print_log


##########################################################################################
class eval_gqa():

    def __init__(
            self,
            tier="val",
            scenes="{tier}_sceneGraphs.json",
            questions="{tier}_all_questions.json",
            choices="{tier}_choices.json",
            predictions="{tier}_predictions.json",
            attentions="{tier}_attentions.json",
            consistency=False,
            grounding=False,
            objectFeatures=False,
            mapSize=7,
    ):

        self.consistency = consistency
        self.grounding = grounding
        self.objectFeatures = objectFeatures
        self.mapSize = mapSize
        if not consistency:
            print_log("Please consider using --consistency to compute consistency scores for entailed questions.",
                      'current')
            print_log("If you do so, please provide answers to all questions in val_all_questions.json.\n", 'current')

        if not grounding:
            print_log("Please consider using --grounding to compute attention scores.", 'current')
            print_log("If you do so, please provide attention maps through --attentions.\n", 'current')

        ##### Files Loading
        ##########################################################################################
        # Load scene graphs
        print_log("Loading scene graphs...", 'current')
        try:
            self.scenes = self.loadFile(scenes.format(tier=self.tier))
        except:
            print_log('Failed to load scene graphs -- cannot evaluate grounding')
            self.scenes = None  # for testdev

        # Load questions
        print_log("Loading questions...", 'current')
        self.questions = self.loadFile(questions)

        # Load choices
        print_log("Loading choices...", 'current')
        try:
            self.choices = self.loadFile(choices.format(tier=self.tier))
        except:
            print_log('Failed to load choices -- cannot evaluate validity or plausibility', 'current')
            self.choices = None  # for testdev

        # Load predictions and turn them into a dictionary
        print_log("Loading predictions...", 'current')
        predictions = self.loadFile(predictions.format(tier=tier))
        self.predictions = {p["questionId"]: p["prediction"] for p in predictions}

        # Make sure all question have predictions
        for qid in self.questions:
            if (qid not in self.predictions) and (consistency or self.questions[qid]["isBalanced"]):
                print_log("no prediction for question {}. Please add prediction for all questions.".format(qid),
                          'current')
                raise Exception("missing predictions")

        # Load attentions and turn them into a dictionary
        self.attentions = None
        if grounding:
            with open(attentions.format(tier=tier)) as attentionsFile:
                attentions = json.load(attentionsFile)
                self.attentions = {a["questionId"]: a["attention"] for a in attentions}

    def forward(self):
        # Initialize data structure to track all metrics: e.g. accuracy, validity and plausibility, as well as
        # accuracy per question type, length and number of reasoning steps.
        scores = {
            "accuracy": [],  # list of accuracies per question (1 if correct else 0). Will be averaged ultimately.
            "binary": [],
            # list of accuracies per a binary question (1 if correct else 0). Will be averaged ultimately.
            "open": [],  # list of accuracies per an open question (1 if correct else 0). Will be averaged ultimately.
            "validity": [],  # list of validity per question (1 if valid else 0).
            "plausibility": [],  # list of plausibility per question (1 if plausible else 0).
            "consistency": [],  # list of consistency scores for entailed questions.
            "accuracyPerStructuralType": defaultdict(list),
            # list of question accuracies for each structural type (e.g. compare, logic questions).
            "accuracyPerSemanticType": defaultdict(list),
            # list of question accuracies for each semantic type (e.g. questions about an object, an attribute, a relation).
            "accuracyPerLength": defaultdict(list),  # list of question accuracies per question's word number.
            "accuracyPerSteps": defaultdict(list),
            # list of question accuracies per question's reasoning length (steps number).
            "grounding": [],  # list of grounding scores for each question.
        }

        # Initialize golden and predicted histograms per each question group. Used to compute the distribution metric.
        dist = {"gold": defaultdict(lambda: defaultdict(int)), "predicted": defaultdict(lambda: defaultdict(int))}
        ##### Main score computation
        ##########################################################################################

        # Loop over the questions and compute mterics
        for qid, question in tqdm(self.questions.items()):

            # Compute scores over the balanced dataset (more robust against cheating by making educated guesses)
            if question["isBalanced"]:
                gold = question["answer"]
                predicted = self.predictions[qid]

                correct = predicted == gold
                score = self.toScore(correct)

                wordsNum = self.getWordsNum(question)
                stepsNum = self.getStepsNum(question)

                # Update accuracy
                scores["accuracy"].append(score)
                scores["accuracyPerLength"][wordsNum].append(score)
                scores["accuracyPerSteps"][stepsNum].append(score)
                scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
                scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
                answerType = "open" if question["types"]["structural"] == "query" else "binary"
                scores[answerType].append(score)

                # Update validity score
                valid = (
                    self.belongs(predicted, self.choices[qid]["valid"], question) if self.choices else False
                )
                scores["validity"].append(self.toScore(valid))

                # Update plausibility score
                plausible = (
                    self.belongs(predicted, self.choices[qid]["plausible"], question)
                    if self.choices
                    else False
                )
                scores["plausibility"].append(self.toScore(plausible))

                # Optionally compute grounding (attention) score
                if self.attentions is not None:
                    groundingScore = self.computeGroundingScore(
                        question, self.scenes[question["imageId"]], self.attentions[qid]
                    )
                    if groundingScore is not None:
                        scores["grounding"].append(groundingScore)

                # Update histograms for gold and predicted answers
                globalGroup = question["groups"]["global"]
                if globalGroup is not None:
                    dist["gold"][globalGroup][gold] += 1
                    dist["predicted"][globalGroup][predicted] += 1

                if self.consistency:
                    # Compute consistency (for entailed questions)
                    scores = self.updateConsistency(qid, question, self.questions, correct, scores)

        # Compute distribution score
        scores["distribution"] = self.chiSquare(dist["gold"], dist["predicted"]) / 100

        # Average scores over all questions (in the balanced dataset) and print_log scores

        metrics = [
            "binary",
            "open",
            "accuracy",
            "consistency",
            "validity",
            "plausibility",
            "grounding",
            "distribution",
        ]

        detailedMetrics = [
            ("accuracyPerStructuralType", "Accuracy / structural type"),
            ("accuracyPerSemanticType", "Accuracy / semantic type"),
            ("accuracyPerSteps", "Accuracy / steps number"),
            ("accuracyPerLength", "Accuracy / words number"),
        ]

        subMetrics = {"attr": "attribute", "cat": "category", "global": "scene", "obj": "object", "rel": "relation"}
        # average
        for k in metrics:
            if isinstance(scores[k], list):
                scores[k] = self.avg(scores[k]) * 100

        for k, _ in detailedMetrics:
            for t in scores[k]:
                scores[k][t] = self.avg(scores[k][t]) * 100, len(scores[k][t])

        # print_log
        for m in metrics:
            # skip grounding and consistency scores if not requested
            if m == "grounding" and not self.grounding:
                continue
            if m == "consistency" and not self.consistency:
                continue

            # print_log score
            print_log(
                "{title}: {score:.2f}{suffix}".format(
                    title=m.capitalize(),
                    score=scores[m],
                    suffix=" (lower is better)" if m == "distribution" else "%",
                )
                , 'current')

        for m, mPrintName in detailedMetrics:
            print_log("")
            # print_log metric title
            print_log("{}:".format(mPrintName))

            for t in sorted(list(scores[m].keys())):
                # set sub-metric title
                tName = t
                if isinstance(scores[k], list):
                    tName = subMetrics.get(t, t).capitalize()

                # print_log score
                print_log(
                    "  {title}: {score:.2f}{suffix} ({amount} questions)".format(
                        title=tName, score=scores[m][t][0], suffix="%", amount=scores[m][t][1]
                    )
                    , 'current')
        return scores

    def loadFile(self, name):
        # load standard json file
        if os.path.isfile(name):
            with open(name) as file:
                data = json.load(file)
        # load file chunks if too big
        elif os.path.isdir(name.split(".")[0]):
            data = {}
            chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir=name.split(".")[0], ext=name.split(".")[1]))
            for chunk in chunks:
                with open(chunk) as file:
                    data.update(json.load(file))
        else:
            raise Exception("Can't find {}".format(name))
        return data

    ##### Scores data structures initialization
    ##########################################################################################

    # book to float
    def toScore(self, b):
        return float(1 if b else 0)

    # Compute average of a list
    def avg(self, l):
        if len(l) == 0:
            return 0
        return float(sum(l)) / len(l)

    def wavg(self, l, w):
        if sum(w) == 0:
            return None
        return float(sum(l[i] * w[i] for i in range(len(l)))) / sum(w)

    ##### Question lengths - words numbers and reasoning steps number
    ##########################################################################################

    # Compute question length (words number)
    def getWordsNum(self, question):
        return len(question["question"].split())

    # Compute number of reasoning steps (excluding the final "querying" step which doesn't increase effective reasoning length)
    def getStepsNum(self, question):
        return len(
            [
                c
                for c in question["semantic"]
                if not (
                any(
                    [
                        o in "{}: {}".format(c["operation"], c["argument"])
                        for o in ["exist", "query: name", "choose name"]
                    ]
                )
            )
            ]
        )

    ##### Functions for question annotations
    ##########################################################################################

    # # Utility function for converting question annotations string keys to slices
    # def toSlice(strSlice):
    #     sliceLims = (int(n) for n in strSlice.split(':'))
    #     return apply(slice, sliceLims)

    # # Utility function for converting question annotations string keys to indexes list:
    # # "1" => [0]
    # # "1:3" => [1, 2]
    # # "4:9:2" => [4, 6, 8]
    # def intsFromSlice(strSlice):
    #     slice_obj = get_slice_obj(slicearg)
    #     return range(slice_obj.start or 0, slice_obj.stop or -1, slice_obj.step or 1)

    ##### Functions for validity and plausibility
    ##########################################################################################

    def belongs(self, element, group, question):
        # normalization ()
        if "Common" in question["types"]["detailed"]:
            group = ["color", "material", "shape"]

        return element in group

    ##### Functions for consistency scores (for entailed questions ("inferred"))
    ##########################################################################################

    def updateConsistency(self, questionId, question, questions, correct, scores):
        inferredQuestions = [eid for eid in question["entailed"] if eid != questionId]

        if correct and len(inferredQuestions) > 0:

            cosnsitencyScores = []
            for eid in inferredQuestions:
                gold = questions[eid]["answer"]
                predicted = self.predictions[eid]
                score = self.toScore(predicted == gold)
                cosnsitencyScores.append(score)

            scores["consistency"].append(self.avg(cosnsitencyScores))
        return scores

    ##### Functions for grounding score (optional, only for attention models)
    ##########################################################################################

    # Utility functions for working with bounding boxes.
    # c = (x0, y0, x1, y1), r = (r0, r1)

    def yrange(self, c):
        return (c[1], c[3])

    def xrange(self, c):
        return (c[0], c[2])

    def length(self, r):
        if r is None:
            return 0
        return float(r[1] - r[0])

    def size(self, c):
        return self.length(self.xrange(c)) * self.length(self.yrange(c))

    def intersection(self, r1, r2):
        ir = (max(r1[0], r2[0]), min(r1[1], r2[1]))
        if ir[1] > ir[0]:
            return ir
        return None

    def intersectionSize(self, c1, c2):
        return self.length(self.intersection(self.xrange(c1), self.xrange(c2))) * self.length(
            self.intersection(self.yrange(c1), self.yrange(c2))
        )

    def intersectionRate(self, c1, c2):
        return float(self.intersectionSize(c1, c2)) / self.size(c1)

    # Get spatial cell
    def getCell(self, i, j):
        edge = float(1) / self.mapSize
        return (edge * i, edge * j, edge * (i + 1), edge * (j + 1))

    # Get bounding box of objectId in sceneGraph
    def getRegion(self, sceneGraph, objectId):
        obj = sceneGraph["objects"][objectId]
        x0 = float(obj["x"]) / sceneGraph["width"]
        y0 = float(obj["y"]) / sceneGraph["height"]
        x1 = float(obj["x"] + obj["w"]) / sceneGraph["width"]
        y1 = float(obj["y"] + obj["h"]) / sceneGraph["height"]
        return (x0, y0, x1, y1)

    # Compute grounding score. Computer amount of attention (probability) given to each of the regions
    # the question and answers refer to.
    def computeGroundingScore(self, question, sceneGraph, attentionMap):
        ## prepare gold regions
        regions = []
        # add question regions
        regions += [
            self.getRegion(sceneGraph, pointer) for pointer in question["annotations"]["question"].values()
        ]
        # add answer regions
        regions += [
            self.getRegion(sceneGraph, pointer) for pointer in question["annotations"]["fullAnswer"].values()
        ]
        # add all the image if the question refers to the whole scene
        if any(("scene" in c) for c in question["semantic"]):
            regions.append((0, 0, 1, 1))

        # prepare attention map
        if self.objectFeatures:
            # cells = [((x0, y0, x1, y1), attention) for x0, y0, x1, y1, attention in cells]
            pass
        else:
            cells = [
                (self.getCell(i, j), attentionMap[i][j])
                for i in range(self.mapSize)
                for j in range(self.mapSize)
            ]

        # compare attention map to gold regions
        scores = []
        for region in regions:
            for cell, attention in cells:
                scores.append(attention * self.intersectionRate(cell, region))
        return sum(scores)

    ##### Functions for distribution score
    ##########################################################################################

    # Compute chi square statistic of gold distribution vs predicted distribution,
    # averaged over all question groups
    def chiSquare(self, goldDist, predictedDist):
        sumScore, sumOverall = 0, 0

        for group in goldDist:
            score, overall = 0, 0

            for ans in goldDist[group]:
                e = goldDist[group][ans]
                o = predictedDist[group].get(ans, 0)
                score += (float(o - e) ** 2) / e
                overall += goldDist[group][ans]

            sumScore += score * overall
            sumOverall += overall

        avgScore = float(sumScore) / sumOverall

        return avgScore
