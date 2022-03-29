import json

exclude = ['gay', 'straight', 'lesbian']

def calculateResults(filename:str):
    with open(filename, 'r') as f:
        data = json.load(f)
        results = {}
        for attributeType in data:
            results[attributeType] = {}
            for attribute in data[attributeType]:
                lowestIndex = 30500
                sexType = 'none'
                for sexualityType, value in data[attributeType][attribute].items():
                    # if sexualityType != 'gay man':
                    # print(sexualityType)
                    # if sexualityType in exclude:
                    #     continue
                    if sexualityType == 'none':
                        continue
                    if value['index']:
                        index = value['index']
                        if lowestIndex > index:
                            lowestIndex = index
                            sexType = sexualityType
                            weight = float(value['weight'])
                # if attribute == 'Aggressive':
                #     print(attribute, sexType, index, data[attributeType][attribute])
                if sexType in results[attributeType]:
                    if attribute not in results[attributeType][sexType]:
                        results[attributeType][sexType].append(attribute)
                else:
                    results[attributeType][sexType] = []
        with open(f"data/final_{filename.split('/')[1]}", 'w') as f:
            f.write(json.dumps(results, indent=4))

calculateResults('data/results_attribute_check_2_30500.json')
calculateResults('data/results_attribute_check_3_30500.json')
calculateResults('data/results_attribute_check_5_30500.json')
# calculateResults('data/results_attribute_check_12_30500.json')


def countAmountAttributes(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)
        for attributeType in data:
            for sexualOrientation in data[attributeType]:
                # print(sexualOrientation)
                if sexualOrientation != 'none':
                    bestAttributes = data[attributeType][sexualOrientation]
                    amount = len(bestAttributes)
                    print(f'{attributeType:8s}: {sexualOrientation:15s} {amount}')
            print()


# countAmountAttributes('data/final_results_attribute_check_12_30500.json')
countAmountAttributes('data/final_results_attribute_check_5_30500.json')
countAmountAttributes('data/final_results_attribute_check_3_30500.json')
countAmountAttributes('data/final_results_attribute_check_2_30500.json')


print("straight pos", (100/142)*117)
print("gay pos", (100/142)*10)
print("lesbian pos", (100/142)*11)
print()
print("straight neg", (100/157)*112)
print("gay neg", (100/157)*16)
print("lesbian neg", (100/157)*23)
print()
print()
print("hetero pos", (100/136)*45)
print("homo pos", (100/136)*58)
print("bi pos", (100/136)*8)
print("a pos", (100/136)*17)
print("pan pos", (100/136)*8)
print()
print("hetero neg", (100/149)*23)
print("homo neg", (100/149)*55)
print("bi neg", (100/149)*7)
print("a neg", (100/149)*39)
print("pan neg", (100/149)*25)

