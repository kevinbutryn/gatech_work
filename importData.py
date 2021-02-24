# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

two = {
    "Private":1, 
    "Self-emp-not-inc":2, 
    "Self-emp-inc":3, 
    "Federal-gov":4, 
    "Local-gov":5, 
    "State-gov":6, 
    "Without-pay":7, 
    "Never-worked":8
}
four = {
    "Bachelors":1, 
    "Some-college":2, 
    "11th":3, 
    "HS-grad":4, 
    "Prof-school":5, 
    "Assoc-acdm":6, 
    "Assoc-voc":7, 
    "9th":8, 
    "7th-8th":9, 
    "12th":10, 
    "Masters":11, 
    "1st-4th":12, 
    "10th":13, 
    "Doctorate":14, 
    "5th-6th":15, 
    "Preschool":16
}
six = {
    "Married-civ-spouse":1, 
    "Divorced":2, 
    "Never-married":3, 
    "Separated":4, 
    "Widowed":5, 
    "Married-spouse-absent":6, 
    "Married-AF-spouse":7
}
seven = {
    "Tech-support":1, 
    "Craft-repair":2, 
    "Other-service":3, 
    "Sales":4, 
    "Exec-managerial":5, 
    "Prof-specialty":6, 
    "Handlers-cleaners":7, 
    "Machine-op-inspct":8, 
    "Adm-clerical":9, 
    "Farming-fishing":10, 
    "Transport-moving":11, 
    "Priv-house-serv":12, 
    "Protective-serv":13, 
    "Armed-Forces":14
}
eight = {
    "Wife":1, 
    "Own-child":2, 
    "Husband":3, 
    "Not-in-family":4, 
    "Other-relative":5, 
    "Unmarried":6
}
nine = {
    "White":1, 
    "Asian-Pac-Islander":2, 
    "Amer-Indian-Eskimo":3, 
    "Other":4, 
    "Black":5
}
ten = {
    "Female":1, 
    "Male":2, 
}
fourteen = {
    "United-States":1,
    "Cambodia":2,
    "England":3,
    "Puerto-Rico":4,
    "Canada":5,
    "Germany":6,
    "Outlying-US(Guam-USVI-etc)":7,
    "India":8,
    "Japan":9,
    "Greece":10,
    "South":11,
    "China":12,
    "Cuba":13,
    "Iran":14,
    "Honduras":15,
    "Philippines":16,
    "Italy":17,
    "Poland":18,
    "Jamaica":19,
    "Vietnam":20,
    "Mexico":21,
    "Portugal":22,
    "Ireland":23,
    "France":24,
    "Dominican-Republic":25,
    "Laos":26,
    "Ecuador":27,
    "Taiwan":28,
    "Haiti":29,
    "Columbia":30,
    "Hungary":31,
    "Guatemala":32,
    "Nicaragua":33,
    "Scotland":34,
    "Thailand":35,
    "Yugoslavia":36,
    "El-Salvador":37,
    "Trinadad&Tobago":38,
    "Peru":39,
    "Hong":40,
    "Holand-Netherlands":41
}
fifteen = {
    ">50K.\n":'0\n', 
    "<=50K.\n":'1\n', 
}

#make dictionary
# splits = s.split(",")
# x = 1
# for thing in splits:
#     print('"'+thing+'":'+str(x)+",")
#     x = x +1




# f = open('adult.test', "r")
# f2 = open("modified_test_orig", "a")

f = open('adult.data', "r")
f2 = open("modified_data_orig", "a")

for line in f:

    if '?' in line:
        continue

    # parts = line.split(',')
    # newline = parts[0] + ',' + str(two[parts[1]]) + ',' + parts[2] + ',' + str(four[parts[3]]) + ',' + parts[4] + ',' + str(six[parts[5]]) + ',' + str(seven[parts[6]]) + ',' + str(eight[parts[7]]) + ',' + str(nine[parts[8]]) + ',' + str(ten[parts[9]]) + ',' + parts[10] + ',' + parts[11] + ',' + parts[12] + ',' + str(fourteen[parts[13]]) +',' + str(fifteen[parts[14]])
    # print(line)
    # print(newline)
    # f2.write(newline)
    f2.write(line)

f.close()
f2.close()