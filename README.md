# Introduction
This was my final project for Data Science with Python. My task is to use different machine learning methods to see which one produces the highest accuracy.

The dataset that I picked to talk about is about League of Legends. This dataset had both binary and nominal data. A little bit about the game is that it is a game of five players versus another group of five players. The way to win the game is to capture the enemy Nexus. There are three paths to the nexus protected by two turrets. 
	When it came to setting up the dataset since it was already in csv format I did not need to do any additional formatting to get it into python. There were also no empty spaces or N/A values that would become an issue later on. The original dataset had around 30k rows and almost 50 columns, and rather than using all of the data set I selected certain columns that I want to focus on. I made sure to select columns that were related to winning but not directly related to winning like turrets destroyed. 

# Results
##KNN
For my first classification, I’m going to be using k-nn. The first step that I did was to find the highest accuracy K and I found out that the highest K is when k=4. When I plug it in and compared it to the test I found out that the machine was more likely to predict a false negative than a false positive. 

When it came to feature selection, I first looked at features selection with 4 features, there were two results 0.727 and 0.729. The ones with 0.729 all have wards placed as one of the features. I thought it would be really interesting to test out three features selection. So I kept wards placed and replaced the other two features. I found out that that the highest accuracy rate came from selecting three features of ward placed, kills, and average level with an accuracy rate of 83.8%
