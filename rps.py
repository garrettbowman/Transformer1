
# generate  a rock paper scissors game
import random
# generate a random integer between 1 and 3 inclusive
computer_choice = random.randint(1,3)   
# prompt the user to enter a value
user_choice = input('Please enter your choice: ')

# convert the user choice to an integer
user_choice = int(user_choice)

# print the computer choice
print(computer_choice)

# print the user choice
print(user_choice)

# print the winner
if computer_choice == user_choice:
    print('Tie')
elif computer_choice == 1 and user_choice == 2:
    print('User wins')
elif computer_choice == 1 and user_choice == 3:
    print('Computer wins')
elif computer_choice == 2 and user_choice == 1:
    print('Computer wins')
elif computer_choice == 2 and user_choice == 3:
    print('User wins')
elif computer_choice == 3 and user_choice == 1:
    print('User wins')
elif computer_choice == 3 and user_choice == 2:
    print('Computer wins')
else:
    print('Invalid choice')
