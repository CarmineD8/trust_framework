#!/usr/bin/env python3

import sys,random
from random import choice
import sys  

from ebcic import Params, exact, verify_interval_of_p, print_interval

        
def main(args):

    """Print exact interval as text.
    Edit the following parameters, k, n, confi_perc, and run this cell.
    """
    print_interval(Params(
        k=4,          # Number of errors
        n=7,          # number of trials
        confi_perc = 90  # Confidence percentage (0-100)
    ))
    
    str1, str2 = exact(Params(
        k=4,          # Number of errors
        n=7,          # number of trials
        confi_perc = 90  # Confidence percentage (0-100)
    ))
    if verify_interval_of_p(Params(
        k=4,          # Number of errors
        n=7,          # number of trials
        confi_perc = 90  # Confidence percentage (0-100)
    ),str1, str2, -14, verbose=2) != 0:
        print("UNRELIABLE")
    else:
        print("RELIABLE")
    print(str(str1) + " " + str(str2))
    # ebcic.interval_graph(ebcic.GraProps(
    # k_start=0,  # >= 0
    # k_end=5,    # > k_start
    # line_list=['with_exact']))
    # ebcic.interval_graph(ebcic.GraProps(
    # # Set the range of k with k_*
    # k_start=1,  # >= 0
    # k_end=1,    # > k_start
    # k_step=1,   # > 0
    # # Edit the list of confidence percentages (0-100):
    # confi_perc_list=[90, 95, 99, 99.9, 99.99],
    # # Lines to depict

    # line_list=[
    #     'with_exact',
    #     'with_line_kn',  # Line of k/n
    # ],
    # ))

if __name__ == "__main__":
    main(sys.argv)




    # def getFixedRates(message):
#     flag_not_float = False
#     rate = "-1.0"
#     try:
#         rate = float(input(message))
#     except ValueError:
#         flag_not_float = True

#     while (float(rate) <0.0 or float(rate) > 1.0 or flag_not_float):
#         try:                    
#             flag_not_float = False
#             rate = float(input("ERROR. " + message))
#         except ValueError:
#             flag_not_float = True

#     return rate 

# def getTimeLength(message):
#     flag_not_float = False
#     rate = "-1.0"
#     try:
#         rate = float(input(message))
#     except ValueError:
#         flag_not_float = True

#     while (float(rate) <0.0 or flag_not_float):
#         try:                    
#             flag_not_float = False
#             rate = float(input("ERROR. " + message))
#         except ValueError:
#             flag_not_float = True

#     return rate 

# def getFixedAgentsID(event_list, agents_count):
#     fixed_rates_agent_list = list()
#     check = None
#     flag_not_int = False

#     for x in event_list:
#         for y in x.related_actions:
#             print(fixed_rates_agent_list)
#             check = -1
#             while check != 0:
#                 try:
#                     check = int(input(" \n Do you want to fix the rates of an agent for the action %s during event %s?: \n Enter 1 for G1, 2 for G2, ..., %d for G%d. \n Enter 0 to exit.\n" %(y.ID, x.ID, agents_count, agents_count)))
#                 except ValueError:
#                     flag_not_int = True
                
#                 while (flag_not_int or check not in range(0,agents_count+1)):
#                     try:                    
#                         flag_not_int = False
#                         check = int(input(" \n ERROR.\n Enter 1 for G1, 2 for G2, ..., %d for G%d. \n Enter 0 to exit.\n" %(agents_count, agents_count)))
#                     except ValueError:
#                         flag_not_int = True
                    
#                 if [check, y.ID, x.ID] not in fixed_rates_agent_list and check != 0:
#                         fixed_rates_agent_list.append([check,y.ID, x.ID]) 


#     return fixed_rates_agent_list