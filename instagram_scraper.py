# need to install instagram-scraper (pip install instagram-scraper)

#from InstagramAPI import InstagramAPI
from __future__ import print_function
import fnmatch
import os
import subprocess
import json
import Queue
from shutil import copyfile
from multiprocessing import Process
import requests


def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

def get_num_random_user_comments(close_user, random_user, min_num_words):
	try:	
		json_file = open("./" + close_user + "/" + close_user + ".json")
		json_data = json.loads(json_file.read())
		json_data = json_data["GraphImages"]
		num_random_user_comments = 0
		for data in json_data:
			comments = data["comments"]["data"]
			for comment in comments:
				words = comment["text"].strip().split(' ')
				words = list(filter(lambda x: len(x) > 0, words))
				# if comment["owner"]["username"] == random_user:
				# 	print comment["text"]
				# 	print len(words)
				# 	raw_input()
				if comment["owner"]["username"] == random_user and len(words) >= min_num_words:
					num_random_user_comments += 1
					break # Count only one comment per post
	except:
		return 0
	return num_random_user_comments


def get_close_users(random_user):
	assert os.path.exists("./" + random_user + "/" + random_user + ".json")

	json_file = open("./" + random_user + "/" + random_user + ".json")
	json_data = json.loads(json_file.read())
	json_data = json_data["GraphImages"]
	close_users = set([])
	for data in json_data:
		comments = data["comments"]["data"]
		for comment in comments:
			if not comment["owner"]["username"] == random_user:
				close_users.add(comment["owner"]["username"])
	return list(close_users)


def insta_json_scrap(user):
	if not os.path.exists("./" + user):
		os.system("instagram-scraper " + user + " -u a -p b --comments -t none -T {shortcode} --retry-forever")
		print ('')


def insta_scrap(user):
	#si.wShowWindow = subprocess.SW_HIDE # default
	#if not os.path.exists("./" + user):
	print (user + " processing..")
	os.system("instagram-scraper " + user + " -u a -p b --comments -T {shortcode} --retry-forever -q > dummy.txt")
	#subprocess.call("instagram-scraper " + user + " -u a -p b --comments -T {shortcode} --retry-forever -q > dummy.txt")
	print (user + " finished")
	#print '' 


def filter_images(random_user, close_user, min_num_words):
	assert os.path.exists("./" + close_user + "/" + close_user + ".json")

	copied = False
	json_file = open("./" + close_user + "/" + close_user + ".json")
	try:
		json_data = json.loads(json_file.read())
	except:
		return
	json_data = json_data["GraphImages"]

	for data in json_data:
		isDeleted = True
		comments = data["comments"]["data"]
		file_name = data["shortcode"]

		for comment in comments:
			words = comment["text"].strip().split(' ')
			words = list(filter(lambda x: len(x) > 0, words))
			if comment["owner"]["username"] == random_user and len(words) >= min_num_words:
				print ("File " + file_name + " selected")
				#os.system("rm -rf ./" + close_user + "/" + file_name + "*")
				if not os.path.exists("./final_dataset/" + close_user):
					os.makedirs("./final_dataset/" + close_user)
				if os.path.exists("./" + close_user + "/" + file_name + ".jpg"):
					copyfile("./" + close_user + "/" + file_name + ".jpg", "./final_dataset/" + close_user + "/" + file_name + ".jpg")
					copied = True

	if copied:
		if not os.path.exists("./final_dataset/" + close_user):
			os.makedirs("./final_dataset/" + close_user)
		copyfile("./" + close_user + "/" + close_user + ".json", "./final_dataset/" + close_user + "/" + close_user + ".json")
			


	# assert os.path.exists("./" + close_user + "/" + close_user + ".json")

	# json_file = open("./" + close_user + "/" + close_user + ".json")
	# json_data = json.loads(json_file.read())
	# json_data = json_data["GraphImages"]
	# for data in json_data:
	# 	isDeleted = True
	# 	comments = data["comments"]["data"]
	# 	file_name = data["shortcode"]

	# 	for comment in comments:
	# 		words = comment["text"].split(' ')
	# 		words = list(filter(lambda x: len(x) > 0, words))
	# 		if comment["owner"]["username"] == random_user and len(words) >= min_num_words:
	# 			isDeleted = False
	# 			saved_images[file_name] = True
	# 			break
	# 	if isDeleted and not file_name in saved_images:
	# 		file_name = data["shortcode"]
	# 		print "File " + file_name + " of " + close_user + " was deleted"
	# 		os.system("rm -rf ./" + close_user + "/" + file_name + "*")


def delete_all_images(close_users):
	for close_user in close_users:
		if os.path.exists("./" + close_user + "/" + close_user + ".json"):
			json_file = open("./" + close_user + "/" + close_user + ".json")
			json_data = json.loads(json_file.read())
			json_data = json_data["GraphImages"]
			print ("All files of " + close_user + " were deleted")
			os.system("rm -rf ./" + close_user)
			# for data in json_data:
			# 	file_name = data["shortcode"]

				# if not file_name in saved_images:
				# 	os.system("rm -rf ./" + close_user + "/" + file_name + "*")

def num_comments_oneself(random_user):
	assert os.path.exists("./" + random_user + "/" + random_user + ".json")

	num_comments = 0

	json_file = open("./" + random_user + "/" + random_user + ".json")
	json_data = json.loads(json_file.read())
	json_data = json_data["GraphImages"]
	for data in json_data:
		comments = data["comments"]["data"]
		for comment in comments:
			if comment["owner"]["username"] == random_user:
				num_comments += 1

	return num_comments


def get_num_posts(user):
	try:
		response = requests.get('https://www.instagram.com/' + user + '/')
	except:
		return -1
	 
	html = response.text

	start = html.find("edge_owner_to_timeline_media")
	start = html[start:].find("{") + start
	start = html[start:].find("count") + start
	start = html[start:].find(":") + 1 + start

	end = min(html[start:].find(","), html[start:].find("}"))
	try:
		int(html[start:start+end])
	except:
		return -1

	return int(html[start:start+end])


saved_images = {}

max_num_posts = 200
min_num_words = 3
min_num_comments = 10
min_num_comments_oneself = 10

# random user queue
user_queue = Queue.Queue()
selected_users = {}

#user_queue.put('im_hwazzi')

user_queue.put('angelaalbarrann')

total_num_comments = 0
data_user = []

while True:
	# Select next random user from queue
	random_user = user_queue.get()
	print ("\nRandom user: " + random_user)
	# Dismiss the user already selected
	if random_user in selected_users:
		continue
	selected_users[random_user] = True

	# Generate json files of close users of the random user
	if not os.path.exists("./" + random_user):
		# Run insta scrap
		#os.system("sleep 10")
		num_posts = get_num_posts(random_user)
		if num_posts == -1 or num_posts > max_num_posts:
			print ("Skip " + random_user)
			continue

		insta_scrap(random_user)

		# If random_user is private user, skip the user
		if len(find_files("./" + random_user, '*.json')) == 0:
			continue

		# If one's comments on oneself are less than threshold, skip the user
		print ("num_comments_oneself: " + str(num_comments_oneself(random_user)))
		if num_comments_oneself(random_user) < min_num_comments_oneself:
			continue

	# get close users ( users that leave a comment on the random user's posts )
	close_users = get_close_users(random_user)
	total_num_random_user_commments = 0

	procs = []
	for close_user in close_users:
		#os.system("sleep 10")
		num_posts = get_num_posts(close_user)
		if num_posts == -1 or num_posts > max_num_posts:
			print ("Skip " + close_user)
			continue
		if not os.path.exists("./" + close_user):
			proc = Process(target=insta_scrap, args=(close_user,))
			procs.append(proc)
			proc.start()
			#insta_scrap(close_user)
		if len(procs) >= 1000:
			for proc in procs:
				if proc.is_alive():
					proc.join()
			procs = []
	for proc in procs:
		if proc.is_alive():
			proc.join()


	print ("Scrap finished")

	for close_user in close_users:
		# Skip private users
		if not os.path.exists("./" + close_user) or len(find_files("./" + close_user, '*.json')) == 0:
			continue

		num_random_user_comments = get_num_random_user_comments(close_user, random_user, min_num_words)
		total_num_random_user_commments += num_random_user_comments

	print ("Total num comments of " + random_user + ": " + str(total_num_random_user_commments))
	
	if total_num_random_user_commments >= min_num_comments:
		data_user.append(random_user)
		total_num_comments += total_num_random_user_commments
		for close_user in close_users:
			# if not os.path.exists("./" + close_user):
			# 	insta_scrap(close_user)

			# private or skipped user
			if not os.path.exists("./" + close_user) or len(find_files("./" + close_user, '*.json')) == 0:
				continue

			## Leave only the posts that the random user made a comment on
			# copy the posts that the random user made a comment on
			filter_images(random_user, close_user, min_num_words)
	# else:
	# 	delete_all_images(close_users)
	print ("** Selected users: " + str(data_user))
	print ("** Total num comments: " + str(total_num_comments))
	with open("output.txt", "w") as f:
		f.write(str(data_user))
		f.write('\n')
		f.write(str(total_num_comments))
		f.write('\n')
	
	if total_num_comments > 500000:
		break

	for close_user in close_users:
		if user_queue.qsize() > 100: break
		if os.path.exists("./" + close_user) and len(find_files("./" + close_user, '*.json')) > 0:
			user_queue.put(close_user)






	# for close_user in close_users:
	# 	if user_queue.qsize() > 100:
	# 		break
	# 	user_queue.put(close_user)


