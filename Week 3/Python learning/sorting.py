data = {
    "users": [
        {"name": "Balen Shah", "score": 88},
        {"name": "Monkey", "score": 95},
        {"name": "John", "score": 72},
        {"name": "Riya", "score": 88}
    ]
}

# Fetch me only score in list
# Sort users by score desc
# Sort users by score desc then name asc
# Print odd numbers only using LIST COMPREHENSION and also print even odd as string for a list of range 5
# Reverse array inplace



# Fetch me only score in list
scores = [user["score"] for user in data["users"]]
print(scores)

# Sort users by score desc
sorted_by_score = sorted(data['users'], key = lambda x: x['score'], reverse=True)
print(sorted_by_score)

# Sort users by score desc then name asc
sorted_by_score_name = sorted(data['users'], key = lambda x: (-x['score'], x['name']))
# sorted_by_score_name = sorted(data['users'], key = lambda x: (-x['score'], x['name']), reverse=True)
print(sorted_by_score_name)


# Print odd numbers only using LIST COMPREHENSION and also print even odd as string for a list of range 5
list = [5,6,78,9,33,1]
odd_num  = [ x for x in list if x%2 != 0 ]
print("odd_num : ", odd_num)

# even_odd = [ "even" if x%2 == 0 else "odd" for x in range(5)]
# print("even_odd : ", even_odd)


# Reverse array inplace
