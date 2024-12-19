from setuptools import find_packages,setup
from typing import List

def getting_req(file_path:str)->list[str]:
    '''
    this function will return the list of requirements
    '''
    Hyphen_e_dot='-e .'
    requirements=[]
    with open(file_path,"r") as file:
        requirements=file.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if Hyphen_e_dot in requirements:
            requirements.remove(Hyphen_e_dot)
    return requirements

setup(
    name="ML project",
    version="0.0.0.1",
    author="awezix",  
    author_email="aawezix@gmail.com",
    packages=find_packages(),
    # install_requires=['numpy','pandas','seaborn']  #for many packages we cannot type all the packages
    install_requires=getting_req('requirements.txt')
)