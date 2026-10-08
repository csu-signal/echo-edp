import csv
import pandas as pd
import json

humanFilenames = []

llmFilenames = []

users = {}
userCount = 0

keys = {}
keyJudges = []

# Open the file in write mode
with open("CRAFT/craftValidationOutputs.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Key", "Judge", "Question", "Index", "User", "UserType", "Response", "ResponseScore"])

    for l in llmFilenames:
        llmdf = pd.read_pickle("CRAFT/LlmResults/" + l)
        for row in llmdf.iterrows():
            score = 0
            index = row[0]
            values = row[1]
            responses = values["answers_list"]
            question = values["question"]
            model = values["model"]
            key = f'{model}_{values["structure_id"]}_{values["turn"]}'
            judge = question[:-1]
            if(not keyJudges.__contains__(f"{key}_{judge}_{question}")):
                keyJudges.append(f"{key}_{judge}_{question}")

                # Do you find the dialogue to be informative enough to take an action and further the task?
                for value in responses:
                    if(value == "Yes"):
                        score = 1
                    elif(value == "No"):
                        score = 0
                    else:
                        score = 0.5

                    print(f"Key:{key} Judge:{judge}, User:{model} question:{question} index:{index} response:{value} score:{score}")
                    writer.writerow([key, judge, question, index, model, "llm", value, score])
            keys[f"{judge}_{index}"] = key
            #print(row)

    for f in humanFilenames:
        with open("CRAFT/FormResults/" + f, mode='r', newline='', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',')

            lastIndex = 0
            currentIndex = 0
            for row in csvreader:
                email = ''
                score = 0
                index = 0
                for header, value in row.items():
                    if(header == "Username"):
                        email = value
                        if(not users.__contains__(email)):
                            userCount += 1
                            users[email] = userCount
                    if(header != "Timestamp" and header != "Username"):
                        combinedKey = header.split(":")[0]
                        question = combinedKey.split(".")[1]
                        index = combinedKey.split(".")[0]
                        judge = question[:-1]
                        key = keys[f"{judge}_{index}"]

                        if(index != lastIndex):
                            lastIndex = index
                            currentIndex = int(index)
                        else:
                            currentIndex += 1
                        #print(f"{question}: {value}")

                        # Do you find the dialogue to be informative enough to take an action and further the task?
                        if(value == "Yes"):
                            score = 1
                        elif(value == "No"):
                            score = 0
                        else:
                            score = 0.5

                        
                        print(f"Key: {key} Judge:{judge} User:{users[email]} question:{question} index:{currentIndex} response:{value} score:{score}")
                        writer.writerow([key, judge, question, currentIndex, users[email], "human", value, score])

with open("CRAFT/userData.json", "w") as f:
    json.dump(users, f, indent=4)