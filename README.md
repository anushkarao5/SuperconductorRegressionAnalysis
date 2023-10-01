# SuperconductorRegressionAnalysis

You can access my colab notebook [here](https://colab.research.google.com/drive/1rvXt8XBbyUkSVo73d0YCBkaSmE9ebSRx?usp=sharing). This notebook provides in-depth explanations and detailed code.

## Table of Contents:
- [Project Objective](#project-objective)
- [Background Information](#background-information)
- Understanding the Data
- Feature Selection
- Non-Neural Network Models
- Neural Network Models
- Conclusions

## Project Objective
The objectives of this project are to:
- Create a model that best predicts the critical temperature of a superconductor based on its material properties
- Compare model performance based on evaluation metrics
- Determine which features are most important in determining critical temperatures

## Background Information: 

  Superconductors are a unique class of materials that efficiently conduct electricity without electrical resistance or heat loss. When an electrical current is passed through conventional materials, some of the flowing electrons collide with the atoms in the material, creating resistance. This resistance leads to dissipation of energy as heat. 
  
  However, when an electrical current is sent through a superconductor, no collisions or resistance occurs. The electrons flow smoothly through the material, so no energy is lost as heat. This property of zero resistance makes superconductors highly desirable in areas like power transmission and distribution. 
  
  Superconductors also display interesting relationships with magnetic fields. Type 1 superconductors repel outside magnetic fields by creating a shield that prevents exterior magnetic lines from entering (Meissner Effect). When a magnet is placed atop a Type 1 superconductor, the magnet will hover in the air because the superconductor’s force field repels the magnet’s magnetic field. This is how hoverboards and levitating (Maglev) trains work!
  
 Type 2 superconductors allow some magnetic fields into their shields without losing their superconducting properties. MRI machines require strong magnetic fields to create high-quality images. Since Type 2 superconductors allow the controlled entry of magnetic fields, they are ideal for creating the magnetic fields needed for MRIs.
 
  What’s the catch? 
  
  Superconductors require extremely low temperatures to reach their superconducting state. The temperature at which a material enters its superconductive state is the critical temperature. For some reference, the highest temperature superconductor (as of now) is a hydrogen sulfide compound which exhibits superconductivity only at -70 degrees Celsius! Since superconductors require extremely low temperatures, readily applying them in technology is challenging. Much of the current research in the field focuses on discovering superconductors at higher temperatures. 



