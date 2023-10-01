# SuperconductorRegressionAnalysis

Access the full colab notebook [here](https://colab.research.google.com/drive/1rvXt8XBbyUkSVo73d0YCBkaSmE9ebSRx?usp=sharing) for all code and in-depth explanations. 

## Table of Contents:
- [Project Objective](#project-objective)
- [Background Information](#background-information)
- [Understanding the Data](#understanding-the-data)
- Feature Selection
- Non-Neural Network Models
- Neural Network Models
- Conclusions

## Project Objective
The objectives of this project are to:
- Create a model that best predicts the critical temperature of a superconductor based on its material properties
- Compare model performance based on evaluation metrics
- Determine which features are most important in determining critical temperatures

## Project Outcomes 
- Developed a Random Forest model that predicted the critical temperature of a superconductor with root mean square error of 9.41 and R^2 value of 0.92
- Found a subset of 13 of 81 features that explained a minimum of 65% of the variability in the target variable for all linear models
- Found a subset of 25 of 81 features that explained a minimum of 65% of the variability in the target variable for all non-neural network models


## Background Information

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors are a unique class of materials that efficiently conduct electricity without electrical resistance or heat loss. When an electrical current is passed through conventional materials, some of the flowing electrons collide with the atoms in the material, creating resistance. This resistance leads to dissipation of energy as heat.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;However, when an electrical current is sent through a superconductor, no collisions or resistance occurs. The electrons flow smoothly through the material, so no energy is lost as heat. This property of zero resistance makes superconductors highly desirable in areas like power transmission and distribution.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors also display interesting relationships with magnetic fields. Type 1 superconductors repel outside magnetic fields by creating a shield that prevents exterior magnetic lines from entering (Meissner Effect). When a magnet is placed atop a Type 1 superconductor, the magnet will hover in the air because the superconductor’s force field repels the magnet’s magnetic field. This is how hoverboards and levitating (Maglev) trains work!

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Type 2 superconductors allow some magnetic fields into their shields without losing their superconducting properties. MRI machines require strong magnetic fields to create high-quality images. Since Type 2 superconductors allow the controlled entry of magnetic fields, they are ideal for creating the magnetic fields needed for MRIs.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;What’s the catch?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Superconductors require extremely low temperatures to reach their superconducting state. The temperature at which a material enters its superconductive state is the critical temperature. For some reference, the highest temperature superconductor (as of now) is a hydrogen sulfide compound which exhibits superconductivity only at -70 degrees Celsius! Since superconductors require extremely low temperatures, readily applying them in technology is challenging. Much of the current research in the field focuses on discovering superconductors at higher temperatures.

## Understanding the Data
- This data was taken from UC Irvine's machine learning repository.
- The superconductor data set contains 21263 superconducting materials.
- There are 81 features representing the material properties of the superconductors. These 81 features are variations of 9 main features: number of elements, atomic mass,  first ionization energy, atomic radius, density, electron affinity, fusion heat, thermal conductivity, and valence. Click here for a brief introduction to these features.

Here is what the first five rows of the data frame look like before feature selection: 

![Alt Text](https://drive.google.com/file/d/1qBANKNSQDCMQ_4t_CiH4xgmLMCqHlJcK/view?usp=drive_link)







