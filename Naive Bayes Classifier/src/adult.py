import os
import re
import random
import math
import string
import time

time1 = time.time()

age = {}
workclass = {}
fnlwgt = {}
education = {}
education_num = {}
marital_status = {}
occupation = {}
relationship = {}
race = {}
sex = {}
capital_gain = {}
capital_loss = {}
hours_per_week = {}
native_country = {}

age_2 = {}
workclass_2 = {}
fnlwgt_2 = {}
education_2 = {}
education_num_2 = {}
marital_status_2 = {}
occupation_2 = {}
relationship_2 = {}
race_2 = {}
sex_2 = {}
capital_gain_2 = {}
capital_loss_2 = {}
hours_per_week_2 = {}
native_country_2 = {}

allkinds = []
# kind_num = 2
kinds = {'<=50K': 0, '>50K': 0}
train_all_num = 32561
test_all_num = 16281
train_num = 0
test_num = 0
# tmp = 0
train_percent = 1.0
print "when train_percent =", train_percent

# Laplace smoothing constant
alpha = 1e-7

trained = []

is_bucket = 1
bucket_num = 10 # [1, 15]
if is_bucket == 1:
	print 'bucket_num =', bucket_num
bucket = [[17, 90],[] ,[12285, 1490400],[] ,[1, 16],[] ,[] ,[] ,[] ,[] ,[0, 99999], [0, 4356], [1, 99], [] ]
# # 0
# age_min = 17
# age_max = 90
# # 2
# fnlwgt_min = 12285
# fnlwgt_max = 1490400
# # 4
# education_num_min = 1
# education_num_max = 16
# #10
# capital_gain_min = 0
# capital_gain_max = 99999
# # 11
# capital_loss_min = 0
# capital_loss_max = 4356
# # 12
# hours_per_week_min = 1
# hours_per_week_max = 99

def add_attrs(attrs, y):
	if is_bucket:
		for i in range(14):
			if ((i == 0) or (i == 2) or (i == 4) or (i == 10) or (i == 11) or (i == 12)):
				if attrs[i].isdigit():
					attrs[i] = str(int(((int(attrs[i]) - bucket[i][0]) / ((bucket[i][1] - bucket[i][0]) / bucket_num))))
	# global number, tmp
	if y == '<=50K':
		if not(age.has_key(attrs[0])):
			age[attrs[0]] = 0
			# number[attrs[0]] = tmp
			# tmp += 1
		age[attrs[0]] += 1
		if not(workclass.has_key(attrs[1])):
			workclass[attrs[1]] = 0
			# number[attrs[1]] = tmp
			# tmp += 1
		workclass[attrs[1]] += 1
		if not(fnlwgt.has_key(attrs[2])):
			fnlwgt[attrs[2]] = 0
			# number[attrs[2]] = tmp
			# tmp += 1
		fnlwgt[attrs[2]] += 1
		if not(education.has_key(attrs[3])):
			education[attrs[3]] = 0
			# number[attrs[3]] = tmp
			# tmp += 1
		education[attrs[3]] += 1
		if not(education_num.has_key(attrs[4])):
			education_num[attrs[4]] = 0
			# number[attrs[4]] = tmp
			# tmp += 1
		education_num[attrs[4]] += 1
		if not(marital_status.has_key(attrs[5])):
			marital_status[attrs[5]] = 0
			# number[attrs[5]] = tmp
			# tmp += 1
		marital_status[attrs[5]] += 1
		if not(occupation.has_key(attrs[6])):
			occupation[attrs[6]] = 0
			# number[attrs[6]] = tmp
			# tmp += 1
		occupation[attrs[6]] += 1
		if not(relationship.has_key(attrs[7])):
			relationship[attrs[7]] = 0
			# number[attrs[7]] = tmp
			# tmp += 1
		relationship[attrs[7]] += 1
		if not(race.has_key(attrs[8])):
			race[attrs[8]] = 0
			# number[attrs[8]] = tmp
			# tmp += 1
		race[attrs[8]] += 1
		if not(sex.has_key(attrs[9])):
			sex[attrs[9]] = 0
			# number[attrs[9]] = tmp
			# tmp += 1
		sex[attrs[9]] += 1
		if not(capital_gain.has_key(attrs[10])):
			capital_gain[attrs[10]] = 0
			# number[attrs[10]] = tmp
			# tmp += 1
		capital_gain[attrs[10]] += 1
		if not(capital_loss.has_key(attrs[11])):
			capital_loss[attrs[11]] = 0
			# number[attrs[11]] = tmp
			# tmp += 1
		capital_loss[attrs[11]] += 1
		if not(hours_per_week.has_key(attrs[12])):
			hours_per_week[attrs[12]] = 0
			# number[attrs[12]] = tmp
			# tmp += 1
		hours_per_week[attrs[12]] += 1
		if not(native_country.has_key(attrs[13])):
			native_country[attrs[13]] = 0
			# number[attrs[13]] = tmp
			# tmp += 1
		native_country[attrs[13]] += 1
	if y == '>50K':
		if not(age_2.has_key(attrs[0])):
			age_2[attrs[0]] = 0
			# number[attrs[0]] = tmp
			# tmp += 1
		age_2[attrs[0]] += 1
		if not(workclass_2.has_key(attrs[1])):
			workclass_2[attrs[1]] = 0
			# number[attrs[1]] = tmp
			# tmp += 1
		workclass_2[attrs[1]] += 1
		if not(fnlwgt_2.has_key(attrs[2])):
			fnlwgt_2[attrs[2]] = 0
			# number[attrs[2]] = tmp
			# tmp += 1
		fnlwgt_2[attrs[2]] += 1
		if not(education_2.has_key(attrs[3])):
			education_2[attrs[3]] = 0
			# number[attrs[3]] = tmp
			# tmp += 1
		education_2[attrs[3]] += 1
		if not(education_num_2.has_key(attrs[4])):
			education_num_2[attrs[4]] = 0
			# number[attrs[4]] = tmp
			# tmp += 1
		education_num_2[attrs[4]] += 1
		if not(marital_status_2.has_key(attrs[5])):
			marital_status_2[attrs[5]] = 0
			# number[attrs[5]] = tmp
			# tmp += 1
		marital_status_2[attrs[5]] += 1
		if not(occupation_2.has_key(attrs[6])):
			occupation_2[attrs[6]] = 0
			# number[attrs[6]] = tmp
			# tmp += 1
		occupation_2[attrs[6]] += 1
		if not(relationship_2.has_key(attrs[7])):
			relationship_2[attrs[7]] = 0
			# number[attrs[7]] = tmp
			# tmp += 1
		relationship_2[attrs[7]] += 1
		if not(race_2.has_key(attrs[8])):
			race_2[attrs[8]] = 0
			# number[attrs[8]] = tmp
			# tmp += 1
		race_2[attrs[8]] += 1
		if not(sex_2.has_key(attrs[9])):
			sex_2[attrs[9]] = 0
			# number[attrs[9]] = tmp
			# tmp += 1
		sex_2[attrs[9]] += 1
		if not(capital_gain_2.has_key(attrs[10])):
			capital_gain_2[attrs[10]] = 0
			# number[attrs[10]] = tmp
			# tmp += 1
		capital_gain_2[attrs[10]] += 1
		if not(capital_loss_2.has_key(attrs[11])):
			capital_loss_2[attrs[11]] = 0
			# number[attrs[11]] = tmp
			# tmp += 1
		capital_loss_2[attrs[11]] += 1
		if not(hours_per_week_2.has_key(attrs[12])):
			hours_per_week_2[attrs[12]] = 0
			# number[attrs[12]] = tmp
			# tmp += 1
		hours_per_week_2[attrs[12]] += 1
		if not(native_country_2.has_key(attrs[13])):
			native_country_2[attrs[13]] = 0
			# number[attrs[13]] = tmp
			# tmp += 1
		native_country_2[attrs[13]] += 1

for i in range(train_all_num):
	if random.uniform(0, 1) < train_percent:
		trained.append(1)
	else:
		trained.append(0)

# read file and train
train_file = open('adult.train', 'r')
train_lists = train_file.readlines()
for i in range(train_all_num):
	if trained[i] == 0:
		continue
	train_num += 1
	train_list = train_lists[i]
	attrs = train_list.split(',')
	allkinds.append(attrs[14][0:-2])
	kinds[attrs[14][0:-2]] += 1
	add_attrs(attrs[0:14], attrs[14][0:-2])

print 'train_num =', train_num

# attr_len = []
# tmp_len = 0
# attr_len.append(tmp_len)

# age_len = len(age)
# tmp_len += age_len
# attr_len.append(tmp_len)

# workclass_len = len(workclass)
# tmp_len += workclass_len
# attr_len.append(tmp_len)

# fnlwgt_len = len(fnlwgt)
# tmp_len += fnlwgt_len
# attr_len.append(tmp_len)

# education_len = len(education)
# tmp_len += education_len
# attr_len.append(tmp_len)

# education_num_len = len(education_num)
# tmp_len += education_num_len
# attr_len.append(tmp_len)

# marital_status_len = len(marital_status)
# tmp_len += marital_status_len
# attr_len.append(tmp_len)

# occupation_len = len(occupation)
# tmp_len += occupation_len
# attr_len.append(tmp_len)

# relationship_len = len(relationship)
# tmp_len += relationship_len
# attr_len.append(tmp_len)

# race_len = len(race)
# tmp_len += race_len
# attr_len.append(tmp_len)

# sex_len = len(sex)
# tmp_len += sex_len
# attr_len.append(tmp_len)

# capital_gain_len = len(capital_gain)
# tmp_len += capital_gain_len
# attr_len.append(tmp_len)

# capital_loss_len = len(capital_loss)
# tmp_len += capital_loss_len
# attr_len.append(tmp_len)

# hours_per_week_len = len(hours_per_week)
# tmp_len += hours_per_week_len
# attr_len.append(tmp_len)

# native_country_len = len(native_country)
# tmp_len += native_country_len
# attr_len.append(tmp_len)

# print attr_len

# single_attr_y = [[0 for i in range(tmp_len)] for j in range(2)]

Py = {}
Py['<=50K'] = 1.0 * kinds['<=50K'] / train_num
Py['>50K'] = 1.0 * kinds['>50K'] / train_num



# predict & eval
Ak = {'<=50K': 0.0, '>50K': 0.0}
Bk = {'<=50K': 0.0, '>50K': 0.0}
Ck = {'<=50K': 0.0, '>50K': 0.0}
p_precission = 0.0
p_recall = 0.0
test_file = open('adult.test', 'r')
test_lists = test_file.readlines()
M = 15.0
num_y_1 = kinds['<=50K'] + M * alpha
num_y_2 = kinds['>50K'] + M * alpha
num_y_1_14 = 1.0
num_y_2_14 = 1.0
for i in range(14):
	num_y_1_14 *= num_y_1
	num_y_2_14 *= num_y_2
for test_list in test_lists:
	test_num += 1
	attrs = test_list.split(',')
	if is_bucket:
		for i in range(14):
			if ((i == 0) or (i == 2) or (i == 4) or (i == 10) or (i == 11) or (i == 12)):
				if attrs[i].isdigit():
					attrs[i] = str(int(((int(attrs[i]) - bucket[i][0]) / ((bucket[i][1] - bucket[i][0]) / bucket_num))))
	###################
	if not(age.has_key(attrs[0])):
		age[attrs[0]] = 0
	if not(workclass.has_key(attrs[1])):
		workclass[attrs[1]] = 0
	if not(fnlwgt.has_key(attrs[2])):
		fnlwgt[attrs[2]] = 0
	if not(education.has_key(attrs[3])):
		education[attrs[3]] = 0
	if not(education_num.has_key(attrs[4])):
		education_num[attrs[4]] = 0
	if not(marital_status.has_key(attrs[5])):
		marital_status[attrs[5]] = 0
	if not(occupation.has_key(attrs[6])):
		occupation[attrs[6]] = 0
	if not(relationship.has_key(attrs[7])):
		relationship[attrs[7]] = 0
	if not(race.has_key(attrs[8])):
		race[attrs[8]] = 0
	if not(sex.has_key(attrs[9])):
		sex[attrs[9]] = 0
	if not(capital_gain.has_key(attrs[10])):
		capital_gain[attrs[10]] = 0
	if not(capital_loss.has_key(attrs[11])):
		capital_loss[attrs[11]] = 0
	if not(hours_per_week.has_key(attrs[12])):
		hours_per_week[attrs[12]] = 0
	if not(native_country.has_key(attrs[13])):
		native_country[attrs[13]] = 0
	if not(age_2.has_key(attrs[0])):
		age_2[attrs[0]] = 0
	if not(workclass_2.has_key(attrs[1])):
		workclass_2[attrs[1]] = 0
	if not(fnlwgt_2.has_key(attrs[2])):
		fnlwgt_2[attrs[2]] = 0
	if not(education_2.has_key(attrs[3])):
		education_2[attrs[3]] = 0
	if not(education_num_2.has_key(attrs[4])):
		education_num_2[attrs[4]] = 0
	if not(marital_status_2.has_key(attrs[5])):
		marital_status_2[attrs[5]] = 0
	if not(occupation_2.has_key(attrs[6])):
		occupation_2[attrs[6]] = 0
	if not(relationship_2.has_key(attrs[7])):
		relationship_2[attrs[7]] = 0
	if not(race_2.has_key(attrs[8])):
		race_2[attrs[8]] = 0
	if not(sex_2.has_key(attrs[9])):
		sex_2[attrs[9]] = 0
	if not(capital_gain_2.has_key(attrs[10])):
		capital_gain_2[attrs[10]] = 0
	if not(capital_loss_2.has_key(attrs[11])):
		capital_loss_2[attrs[11]] = 0
	if not(hours_per_week_2.has_key(attrs[12])):
		hours_per_week_2[attrs[12]] = 0
	if not(native_country_2.has_key(attrs[13])):
		native_country_2[attrs[13]] = 0
	###################
	pred_y_1 = 1.0 * Py['<=50K'] * (age[attrs[0]] + alpha) * (workclass[attrs[1]] + alpha) * (fnlwgt[attrs[2]] + alpha) * (education[attrs[3]] + alpha) * (education_num[attrs[4]] + alpha) * (marital_status[attrs[5]] + alpha) * (occupation[attrs[6]] + alpha) * (relationship[attrs[7]] + alpha) * (race[attrs[8]] + alpha) * (sex[attrs[9]] + alpha) * (capital_gain[attrs[10]] + alpha) * (capital_loss[attrs[11]] + alpha) * (hours_per_week[attrs[12]] + alpha) * (native_country[attrs[13]] + alpha) / num_y_1_14
	pred_y_2 = 1.0 * Py['>50K'] * (age_2[attrs[0]] + alpha) * (workclass_2[attrs[1]] + alpha) * (fnlwgt_2[attrs[2]] + alpha) * (education_2[attrs[3]] + alpha) * (education_num_2[attrs[4]] + alpha) * (marital_status_2[attrs[5]] + alpha) * (occupation_2[attrs[6]] + alpha) * (relationship_2[attrs[7]] + alpha) * (race_2[attrs[8]] + alpha) * (sex_2[attrs[9]] + alpha) * (capital_gain_2[attrs[10]] + alpha) * (capital_loss_2[attrs[11]] + alpha) * (hours_per_week_2[attrs[12]] + alpha) * (native_country_2[attrs[13]] + alpha) / num_y_2_14
	if pred_y_1 < pred_y_2:
		pred_y = '>50K'
	else:
		pred_y = '<=50K'
	if pred_y == attrs[14][0:-2]:
		Ak[pred_y] += 1.0
	else:
		Bk[pred_y] += 1.0
		Ck[attrs[14][0:-2]] += 1.0

accuracy = 1.0 * (Ak['<=50K'] + Ak['>50K']) / test_num
print "accuracy = ", accuracy

p_precission += (Ak['<=50K'] * 1.0 / (Ak['<=50K'] + Bk['<=50K'])) / 2.0
p_precission += (Ak['>50K'] * 1.0 / (Ak['>50K'] + Bk['>50K'])) / 2.0

p_recall += (Ak['<=50K'] * 1.0 / (Ak['<=50K'] + Ck['<=50K'])) / 2.0
p_recall += (Ak['>50K'] * 1.0 / (Ak['>50K'] + Ck['>50K'])) / 2.0

f_measure = 2.0 * p_precission * p_recall / (p_precission + p_recall)

print "p_precission =", p_precission
print "p_recall =", p_recall
print "f_measure =", f_measure
time2 = time.time()
print "time =", time2 - time1

