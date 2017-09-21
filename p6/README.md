# Summary of findings:
+ Females and young males(under age 15) from middle(PClass 2) and upper class(PClass 1) were more likely to survive.
+ The survival rate for the female passengers is higher than that of the male passengers.

# Design:
+ The goal of the visualization is to answer the following: 
	** Who was more likely to survive, males or females? 
	** Did socio-economical class and age affect the survival rate? 
+ The aforementioned factors were used to create the visualization. 
	** The visualization contains two scatter plots, they are the survival rates of the male passengers in the data set and the survival rates of the female passengers in the data set. A scatter plot on top showing the male passengers in the data set respectively. This visualization would allow viewers to examine and find their own stories.
		- In the scatter plots, the x axis is age and the y axis is Pclass.  
		- Each passenger is represented by a rectangle in the scatter plots, the color red means that the passenger survived and the color blue indicates that they did not survive.		
		- The colors red and blue were used for the survived and did not survived categories respectively because an individual can easily distinguish them. 
		- The opacity of the rectangles was set to 20% because several passengers can belong to the same class, be the same age and gender. Solid red indicates that those passengers had a higher chance of surviving and also that there were many passengers that had the same fate and vice versa for the solid blue.
		- Tooltip allows the viewer to the move mouse over the rectangles to see a deatiled summary of the survival rates.

# Feedback:
+ Person 1: 
	- What do you notice in the visualization?
		** The colors are too transparent. 
		** How to read the graph?
	- What do you think is the main takeaway from this visualization?
		** Females and children have a high survival rate. 

+ Person 2: 
	- What do you notice in the visualization?
		** It seems that PClass 3 is the upper class, PClass 2 is the middle class and PClass 1 is the lower class.
	- What questions do you have about the data?
		** Is each rectangle one idividual? Only one individual's information is shown when the mouse is hovered over the rectangle.
	- What do you think is the main takeaway from this visualization?
		** People in the higher socio-economical class seem to have a higher survival rate.

+ Person 3: 
	- What do you notice in the visualization?
		** There seems to be multiple people on the same spot but only information about one person is shown. You should calculate the total for everyone in a particular spot. 
	- What questions do you have about the data?
		** How to read the graph? Which PClass is the upper class? 

# Changes made based on feedback:
+ Added a tip section that helps the viewer read the visualization.
+ Adjusted the order of socio-economic classes shown in the visualization.
+ Adjusted the opacity of the rectangle.
+ Changed the tool tip to reveal information about every individual on a particular spot.
 
 

# Resources:
+ https://www.kaggle.com/c/titanic: the source of dataset used
