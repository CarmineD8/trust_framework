import os
from pickle import TRUE
import sys
import signal
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pylab import rc, subplot2grid, append, figure, ceil, ones, zeros, uniform, plt
import matplotlib.colors as mcolors
import matplotlib.ticker as plticker
import random
from pathlib import Path
import numpy as np
from numpy.core.numeric import True_
from numpy.core.numerictypes import maximum_sctype
from numpy.lib.function_base import percentile
import rospy
import copy 
from datetime import datetime
from threading import RLock
import errno
import socket
from ebcic import Params, exact, verify_interval_of_p

from std_msgs.msg import String
from sim_msg.msg import Initialization, Event_Instance, Auction, Agent_Action, Auction_Declaration, Result_Declaration
from sim_srv.srv import ActionResult

from matplotlib.pylab import append, zeros 

# TERMINAL PRINT SETTINGS (only for 10 agents)
IDENTETOR = False
# SETTINGS
LOG = True
CB_LOG = True
PRINT_TO_TERM = True
TIMEOUT = 30
LOG_PERIOD = 100

MODE = "BCIC"

BCIC_CONFIDENCE = 90

WINDOW_LENGTH = 500  #era 50

BOOT_WINDOW_LENGTH = 500 #era 50

TEST_PHASE_LENGTH = 10000 # era 1000

class Rates(): #Class to store the estimated succ. and verif. rates of an agent 
    def __init__(self,performer_ID, init_success_rate, verification_rates, init_lsr = 0.0, init_usr = 0.0):
        self.performer_ID = performer_ID 
        self.success_rate = init_success_rate
        self.verification_rates = verification_rates
        self.upper_sr = init_usr
        self.lower_sr = init_lsr

class TrustMetric(): #Class to store msg about the experience related to a particular action 
    def __init__(self,action_ID = "None", init_results = [],  init_reliability = 0.0, init_vtw_summ = [], init_trust_worthiness = 0.0, 
                 plot_syncro = 0):
        
        #action id
        self.ID = action_ID
        #reliability metric
        self.results = init_results
        self.reliability = init_reliability

        #verifiation trustworthiness
        self.vtw_summ = init_vtw_summ
        self.ver_trustworthiness = init_trust_worthiness

        #plot data
        self.rel_plt_data = []
        self.vtw_plt_data = []
        if plot_syncro != 0:
            for i in range (0,plot_syncro):
                self.rel_plt_data.append([0, "NONE-" + str(i)])
                self.vtw_plt_data.append(0)

    def store_results(self, result):
        self.results.append(result)

    def update_vtw_summ(self, vtw_n):
        self.vtw_summ.append(vtw_n)

    def update_metrics(self, auction_ID, winner_ID, perceived_outcome, mode):
        w_results = 0.0
        w_vtw_summ = 0.0

        if mode == "BCIC":
            #COMPUTING RELIABILITY
            if len(self.results) == 0:
                self.reliability = 0.0
            else :
                for result in self.results:
                    w_results += result
                self.reliability = float(w_results) / float(len(self.results))
            #COMPUTING VTW
            if len(self.vtw_summ) == 0:
                self.ver_trustworthiness = 0.0
            else:
                for vtw_n in self.vtw_summ:
                    w_vtw_summ += vtw_n
                self.ver_trustworthiness = float(w_vtw_summ) / float(len(self.vtw_summ))
            
            self.rel_plt_data.append([self.reliability, auction_ID +"/"+ winner_ID + "/" + perceived_outcome])
            self.vtw_plt_data.append(self.ver_trustworthiness)

        elif mode == "WINDOW":
            if len(self.results) == 0:
                self.reliability = 0.0
            else :
                #LAST "WINDOW_LENGTH" RESULTS DATA
                if WINDOW_LENGTH > len(self.results):
                    for result in self.results:
                        w_results += result
                    self.reliability = float(w_results) / float(len(self.results))
                else:
                    for i in range(1,WINDOW_LENGTH+1):
                        n = len(self.results)-i
                        w_results += self.results[n]
                    self.reliability = float(w_results) / float(WINDOW_LENGTH)
           
            if len(self.vtw_summ) == 0:
                self.ver_trustworthiness = 0.0
            else:
                #LAST "WINDOW_LENGTH" VTWn DATA
                if WINDOW_LENGTH > len(self.vtw_summ):
                    for vtw_n in self.vtw_summ:
                        w_vtw_summ += vtw_n
                    self.ver_trustworthiness = float(w_vtw_summ) / float(len(self.vtw_summ))
                else:
                    for i in range(1,WINDOW_LENGTH+1):
                        n = len(self.vtw_summ)- i
                        w_vtw_summ += self.vtw_summ[n]

                    self.ver_trustworthiness = float(w_vtw_summ) / float(WINDOW_LENGTH)
                             
            self.rel_plt_data.append([self.reliability, auction_ID +"/"+ winner_ID + "/" + perceived_outcome])
            self.vtw_plt_data.append(self.ver_trustworthiness)

        elif mode == "BOOT":
            if len(self.results) == 0:
                self.reliability = 0.0
            else :
                for result in self.results:
                    w_results += result
                self.reliability = float(w_results) / float(len(self.results))
           
            if len(self.vtw_summ) == 0:
                self.ver_trustworthiness = 0.0
            else:
                for vtw_n in self.vtw_summ:
                    w_vtw_summ += vtw_n
                self.ver_trustworthiness = float(w_vtw_summ) / float(len(self.vtw_summ))
                             
            self.rel_plt_data.append([self.reliability, auction_ID +"/"+ winner_ID + "/" + perceived_outcome])
            self.vtw_plt_data.append(self.ver_trustworthiness)

        elif mode == "TEST":
            if len(self.results) == 0:
                self.reliability = 0.0
            else :
                for result in self.results:
                    w_results += result
                self.reliability = float(w_results) / float(len(self.results))
           
            if len(self.vtw_summ) == 0:
                self.ver_trustworthiness = 0.0
            else:
                for vtw_n in self.vtw_summ:
                    w_vtw_summ += vtw_n
                self.ver_trustworthiness = float(w_vtw_summ) / float(len(self.vtw_summ))
                             
            self.rel_plt_data.append([self.reliability, auction_ID +"/"+ winner_ID + "/" + perceived_outcome])
            self.vtw_plt_data.append(self.ver_trustworthiness)

class Auct_Decl():
    def __init__(self, auction_ID, auctioneer_ID, declarant_ID, rates_list):
        self.auction_ID = auction_ID
        self.auctioneer_ID = auctioneer_ID
        self.declarant_ID = declarant_ID
        self.rates_list = rates_list

class Res_Decl():
    def __init__(self, auction_ID, declarant_ID,performer_ID, perceived_outcome):
        self.auction_ID = auction_ID
        self.declarant_ID = declarant_ID
        self.performer_ID = performer_ID
        self.perceived_outcome = perceived_outcome

class Act_Decl():
    def __init__(self, auction_ID, declarant_ID,performer_ID, state):
        self.auction_ID = auction_ID
        self.declarant_ID = declarant_ID
        self.performer_ID = performer_ID
        self.state = state

class Auction_Data():
    def __init__(self, auction_ID ="None", auctioneer_ID = "None", state = "None", winner = "None", 
                 expected_participants = "", verifying_agents = "", verifier  = False, state_history = "", 
                 auct_decl_timeout = rospy.Time(0.0), act_decl_timeout = rospy.Time(0.0), res_decl_timeout = rospy.Time(0.0)):
        
        self.ID = auction_ID
        self.auctioneer_ID = auctioneer_ID
        self.state = state
        self.expected_participants = expected_participants
        self.winner = winner
        self.verifying_agents = verifying_agents
        self.verifier  = verifier 
        self.result = None
        self.auct_decl_timeout = auct_decl_timeout
        self.act_decl_timeout = act_decl_timeout
        self.res_decl_timeout = res_decl_timeout
        self.state_history = state_history

class TrustAgent(): #Class to model an agent in the trust framework

    def __init__(self,ID, known_agents_list = [], plans_dict = {}, actions_dict = {}, trust_metric_db = {}, 
                 behavior = None, disposition = None, IP = None, port_number = None, random_seed = None): 
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)      
        signal.signal(signal.SIGTERM, self.signal_handler)     
        # Lock initializations 
        self.sig_lock = RLock()
        self.lock_log = RLock()
        self.lock_us = RLock()
        self.lock_adc = RLock()
        self.lock_act = RLock()
        self.lock_vdc = RLock() 
        self.lock_init = RLock()
        ## Initialization ##
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.log_string = dt_string + "\n" + (" AT INITIALIZATION ").center(78,"#") + "\n"
        #self.reload_string = ""
        self.callback_log = ""
        self.log_on = LOG
        self.callback_log_on = CB_LOG
        self.timeout = TIMEOUT
        self.identetor_flag = IDENTETOR 
        self.print_to_term_flag = PRINT_TO_TERM
        self.mode = MODE
        if self.mode == "TEST":
            self.TEST_phase_on = True
        self.adapter_IP = IP
        self.adapter_port_num = port_number

        #initializing Agent ID 
        self.ID = ID
        #List of all the previously encountered agents
        self.known_agents = known_agents_list
        if self.ID not in self.known_agents:
            self.known_agents.append(self.ID) #At least the agent will know itself
        self.known_agents.sort() #alphabetically ordered list 

        #List to store data about new agents
        self.updated_ka_list = copy.deepcopy(self.known_agents)

        #Dictionary of all plans                                    
        self.plans = dict()
        self.plans.update(plans_dict)
        #Dictionary to store data about new plans
        self.updated_plans_dict = dict()
        self.updated_plans_dict.update(self.plans)

        #Dictionary of all action rates data                                   
        self.actions = dict()
        self.actions.update(actions_dict)
        #Dictionary to store data about new actions
        self.updated_actions_dict = dict()
        self.updated_actions_dict.update(self.actions)

        

        #Computing the initialization msg (sent when the agent enter the framework)
        self.init_msg = self.ComputeInitMsg(self.updated_plans_dict)

        #Dictionary to keep track of the observations made by the agent during the auctions
        self.trust_metric_db = dict()
        if trust_metric_db == {}:
            for known_agent in known_agents_list:
                self.trust_metric_db[known_agent] = {}
        else:
            self.trust_metric_db.update(trust_metric_db)
            
        #List of all the auction that are required to be done
        self.auctions_queue = list()

        #Dictionaries to keep track of all the currently active auctions and the changes of their state
        self.current_auctions_state = dict()
        self.next_auctions_state = dict()

        #Dictionary to keep track of auctions that must be serialized
        self.serial_auctions = dict()        

        #Dictionary to keep track of the completed auctions
        self.auction_database = dict()

        #Keep count of the total number of auctions in which the agent has participated (to make coherent plots)
        self.n_tot_auctions = 0

        #Dictionaries to keep track of the declarations received by the other agents
        self.auct_decls = dict()
        self.verif_decls = dict()
        self.action_decls = dict()

        #Control Variables
        self.behavior = behavior
        self.disposition = disposition

        #Flags
        self.pending_update = False
        self.result_flag = False
        self.save_data = False
               
        #DEBUG
        self.counter = 0
        self.counter_log = 0
        self.plot_data_file_path = ""
        self.log_file_path = ""
        self.experiment_data_file_path = ""


        self.SAFE_log("AGENT ID: <%s|%s|%s>" %(self.ID, self.behavior, self.disposition))
        self.SAFE_log("SETTINGS: \n -DECL. TIMEOUT: -%d \n -LOG: -%s \n -CB_LOG: %s \n -MODE: %s \n "
                     %(self.timeout, self.log_on, self.callback_log_on, self.mode))
        
        if self.adapter_IP != None:
            self.SAFE_log("-IMPLEMENTATION:\n\t IP %s; \n\t PORT# %d" %(self.adapter_IP, self.adapter_port_num))

        self.SAFE_log(self.log_known_agents())
        self.SAFE_log(self.log_actions())
        self.SAFE_log(("").center(78,"#") + "\n")


        #Publishers#
        self.init_pub = rospy.Publisher("/init", Initialization, queue_size = 780)#  
        self.auct_decl_pub = rospy.Publisher("/auction_declarations", Auction_Declaration, queue_size = 780 )#  
        self.auction_pub = rospy.Publisher("/auction", Auction, queue_size = 780)#  
        self.action_pub = rospy.Publisher("/action_declarations", Agent_Action, queue_size = 780)#   
        self.res_server_pub = rospy.Publisher("/result_server", Agent_Action, queue_size = 780)
        self.ver_decl_pub = rospy.Publisher("/result_declarations", Result_Declaration, queue_size = 780)#  
        self.logger_pub = rospy.Publisher(("/logger_"+ self.ID), String, queue_size = 780)
        rospy.sleep(1)

        #Subscribers#
        self.init_sub = rospy.Subscriber("/init", Initialization, self.init_callback, queue_size=780)#  
        self.event_sub = rospy.Subscriber("/simulated_event", Event_Instance, self.event_callback, queue_size = 780)#  
        self.auction_sub = rospy.Subscriber("/auction", Auction, self.auction_callback, queue_size = 780)#  
        self.auct_decl_sub = rospy.Subscriber("/auction_declarations", Auction_Declaration, self.auct_decl_callback, queue_size = 780)#  
        self.action_sub = rospy.Subscriber("/action_declarations", Agent_Action, self.action_callback, queue_size = 780)#  
        self.ver_decl_sub = rospy.Subscriber("/result_declarations", Result_Declaration, self.ver_decl_callback, queue_size = 780)#  
        self.logger_sub = rospy.Subscriber(("/logger_" + self.ID), String, self.logger_callback, queue_size = 780)#  
        

        rospy.sleep(1)

        #Services#
        self.action_result = rospy.ServiceProxy('action_result', ActionResult, persistent=True)
        
        rospy.sleep(1)
        # if self.live_plot_on:
        #     self.plot_pub.publish("START")
        self.print_term(self.identetor("%s Sending INIT" %self.ID))
        self.init_pub.publish(self.init_msg)
        rospy.sleep(5)

    def signal_handler(self, sig, frame):
        with self.sig_lock:
            self.plans.update(self.updated_plans_dict) #Update to latest event actions list
            self.known_agents = copy.deepcopy(self.updated_ka_list)  #Update to latest known agent list
            self.actions.update(self.updated_actions_dict)
            self.save_log()
            self.save_plot_data()
            self.plot_graphs(self.experiment_data_file_path)
        sys.exit(0)

    def logger_callback(self,msg):
        self.save_log()
        self.save_plot_data()
        print(self.ID + " FINISHED TO SAVE DATA")
        
    def init_callback(self,msg):
        self.SAFE_callback_logger("> CALLBACK INIT %s START " %(msg.agent_ID))

        new_agent_ID = msg.agent_ID
        new_agent_data = msg.event_actions_data
        if new_agent_ID in self.updated_ka_list:
            self.SAFE_callback_logger("> CALLBACK INIT %s END A.A.K" %(msg.agent_ID))
            return
        else:
            self.pending_update = True
            #Adding the new agent to the known agent list and adding an entry for its rates to all the known actions
            if new_agent_data == "":   
                self.print_term(self.identetor("%s New Agent %s data is empty" %(self.ID, msg.agent_ID)))
                return   
            self.AddNewAgentRates(new_agent_ID, new_agent_data)
            self.init_pub.publish(self.init_msg)
            
            self.SAFE_log("> The new agent [%s] entered the framework" %(new_agent_ID))

                
        self.SAFE_callback_logger("> CALLBACK INIT %s END " %(msg.agent_ID))

    def event_callback(self,msg): #A new Event_Instance may trigger an auction
        #The agents updates to the latest version of the event action and known agents lists
        self.SAFE_callback_logger("> CALLBACK EVENT %s START " %(msg.event_ID))       
        self.SAFE_log("\n\n" + (str(msg.event_counter) + "_" + msg.event_ID).center(78, "=")) 
        
        if msg.event_counter != 0 and msg.event_counter % TEST_PHASE_LENGTH == 0:
                self.SAFE_log("TEST MODE DEACTIVATED %d " %msg.event_counter)
                self.TEST_phase_on = False

        if msg.event_counter != 0 and msg.event_counter % LOG_PERIOD == 0:
            self.save_data = True
        if self.ID == msg.auctioneer_ID:

            self.auctions_queue.append("%s %s %s" %(msg.event_ID,  msg.event_counter, msg.auctioneer_ID))

        self.SAFE_callback_logger("> CALLBACK EVENT %s END " %(msg.event_ID))

    def auction_callback(self, msg): #Auction START-> each involved agent declares its rates. Auction END->the winner performs the action
        #Check if the agent is busy: in such case the agent won't partecipate in the auction
        #when the auction is started, the agent releases a declaration about its competence in the given action
        self.SAFE_callback_logger("> CALLBACK AUCTION %s / %s START " %(msg.auction_ID, msg.state))

        if msg.state == "AD_START": 

            self.SAFE_log("> Auction [%s] declaration phase is started " %(msg.auction_ID) + 
                        "\n> The auctioneer is [" + msg.auctioneer_ID + "]" )   
            self.SAFE_log((" [" + msg.auction_ID + "] " + msg.state).center(78, "_"))   
  
            if self.ID != msg.auctioneer_ID:
                next_state = "1:AD_START"
                next_state_data = [msg.auction_ID, msg.auctioneer_ID]   
                self.auct_decls[msg.auction_ID] = []   
                self.verif_decls[msg.auction_ID] = []
                self.action_decls[msg.auction_ID] = []
                self.StoreNextStateData(msg.auction_ID, next_state, next_state_data) 
            
        elif msg.state == "AD_END":

            if msg.auction_ID in self.current_auctions_state:
                next_state = "3:AD_END"
                if msg.verifiers_string == "":
                    print("%s>>>>>>>>>>>>>>>ERROR. No ver. agents for this action" %self.ID)

                if self.ID in msg.verifiers_string.split(", "):
                    verifier = True 
                else:
                    verifier = False
                    
                next_state_data = [msg.winner_ID, msg.participants_string, msg.verifiers_string, verifier]
                self.StoreNextStateData(msg.auction_ID, next_state, next_state_data) 
                if msg.winner_ID == self.ID:
                    self.print_term(self.identetor("%s [%s]-> Winner!" %(self.ID, msg.auction_ID)))
                
                self.SAFE_log((" [" + msg.auction_ID + "] " + msg.state).center(78, "_") + 
                                "\n> The winner of auction [%s] is [%s]." %(msg.auction_ID, msg.winner_ID) + 
                                "\n> Verifying agents for auction %s: %s" %(msg.auction_ID, msg.verifiers_string))                   
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. No entry for auction [%s] itself at the end of the auction !?!" 
                                %(self.ID, msg.auction_ID))  
            
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Something strange happend with auction %s declaration state" 
                            %(self.ID, msg.auction_ID))

        self.SAFE_callback_logger("> CALLBACK AUCTION %s / %s END " %(msg.auction_ID, msg.state))
        return

    def auct_decl_callback(self,msg): #Stores the Auction Declarations to be evaluated at the end of the auction
        self.SAFE_callback_logger("> CALLBACK AUCTION DECL %s START " %(msg.auction_ID))
        
        rates_list = self.StringToRates(msg.rates_string)
        auct_decl = Auct_Decl(msg.auction_ID, msg.auctioneer_ID, msg.declarant_ID, copy.deepcopy(rates_list))
        if msg.auctioneer_ID == self.ID:
            with self.lock_adc:
                if msg.auction_ID in self.auct_decls:
                    self.auct_decls[msg.auction_ID].append(auct_decl) 
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. No entry for auction [%s]... STRANGE " %(self.ID, auct_decl.auction_ID))

        else: 
            self.SAFE_callback_logger("> CALLBACK AUCTION DECL %s END / NOT AUCTIONER" %(auct_decl.auction_ID)) 
            return

        self.SAFE_callback_logger("> CALLBACK AUCTION DECL %s END " %(auct_decl.auction_ID)) 

    def action_callback(self,msg): #Action START: the agent starts to observe the agent performing the action to verify the result ...
                                    #Action END: the performer declares the end and the result of the action

        self.SAFE_callback_logger("> CALLBACK ACTION %s / %s START  " %(msg.auction_ID, msg.state))
        #When the start of an action is perceived, the agent checks if it can verify the action 

        if msg.state == "READY_for_Act":
            act_decl = Act_Decl(msg.auction_ID, msg.declarant_ID, msg.performer_ID, msg.state)
            if msg.performer_ID == self.ID:
                with self.lock_act:
                    if msg.auction_ID in self.action_decls:
                        self.action_decls[msg.auction_ID].append(act_decl) 
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. No entry for action [%s]... STRANGE " %(self.ID, act_decl.auction_ID))

        elif msg.state == "Act_START":
            if msg.auction_ID in self.current_auctions_state:
                if self.ID == self.current_auctions_state[msg.auction_ID].winner:
                    next_state = "5:Act_PERFORM"
                elif self.current_auctions_state[msg.auction_ID].verifier:
                    next_state = "5:Act_VERIFICATION"
                else:
                    next_state = "5:Wait_Act_END"
                
                self.StoreNextStateData(msg.auction_ID, next_state)
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. No entry found for auction %s" %(self.ID, msg.auction_ID))
     
        elif msg.state == "Act_END":
                #The agent waits set a waiting time for the result declarations for the acution
            if msg.auction_ID in self.current_auctions_state:
                next_state = "6:Act_END"
                self.StoreNextStateData(msg.auction_ID, next_state)
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. No entry found for auction %s" %(self.ID, msg.auction_ID) )

        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Something strange happend with action state of auction %s" %(self.ID, msg.auction_ID) )

        self.SAFE_callback_logger("> CALLBACK ACTION %s / %s END  " %(msg.auction_ID, msg.state))
    
    def ver_decl_callback(self,msg):
        self.SAFE_callback_logger("> CALLBACK VER DECL %s FROM %s START " %(msg.auction_ID, msg.declarant_ID))
        res_decl = Res_Decl(msg.auction_ID, msg.declarant_ID, msg.performer_ID, msg.perceived_outcome)
        with self.lock_vdc:
            if msg.auction_ID in self.verif_decls:
                self.verif_decls[msg.auction_ID].append(res_decl) 
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Ver. declaration received for auction [%s] but I'm not partecipating" 
                                %(self.ID, msg.auction_ID))  

        self.SAFE_callback_logger("> CALLBACK VER DECL %s FROM %s END " %(msg.auction_ID, msg.declarant_ID))

    def run(self):
        # self.counter += 1
        # if self.counter == 50:
        #     self.len_printer()
        #     self.print_everything_about_trust()
        #     self.counter = 0

        # Check to see if the agent has any pending update about known agents, actions or trust metrics
        if self.pending_update:
            self.known_agents = copy.deepcopy(self.updated_ka_list)  
            self.plans.update(self.updated_plans_dict) 
            self.actions.update(self.updated_actions_dict)
            #Update the init msg    
            self.init_msg = self.ComputeInitMsg(self.plans)
            self.pending_update = False
            self.print_term(self.identetor("%s Updated lists" %self.ID))
        

        if self.save_data:
            print(self.ID + " STARTING TO SAVE DATA")
            self.logger_pub.publish("LOG!")
            self.save_data = False

            
        ##AUCTION STARTER##
        self.AuctionStarter()
        
        # Updating the state of the current auctions
        self.UpdateState() 

        ###AUCTION MANAGER###
        for auction in self.current_auctions_state.values():

            if auction.state == "1:AD_START":

                next_state = "2:Wait_AD_END"
                self.StoreNextStateData(auction.ID, next_state)

                auction_data = auction.ID.split("/")
                event_ID = auction_data[0]
                action_ID = auction_data[1]
                #if yes, the agent declares its estimated value of success and verification to the other agents (and the simulator)
                self.log_event_actions(event_ID) 
                if event_ID in self.plans.keys():
                    if action_ID in self.plans[event_ID]:
                        auct_decl = Auction_Declaration()
                        auct_decl.declarant_ID = self.ID
                        auct_decl.auction_ID = auction.ID
                        auct_decl.auctioneer_ID = auction.auctioneer_ID
                        rates_string, decl_data = self.RatesToString(action_ID)
                        auct_decl.rates_string = rates_string
                        if self.adapter_IP != None:
                            data = self.ID + "|DECLARING|" + decl_data
                            self.sendToAdapter(data)
                        self.auct_decl_pub.publish(auct_decl)
                        # self.SAFE_log("> For auction [%s], I have declared %s" %(auct_decl.auction_ID, auct_decl.rates_string))
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. No entry found for action [%s]" %(self.ID, action_ID)) 
                else:    
                    print("%s>>>>>>>>>>>>>>>ERROR. No entry found for event [%s]" %(self.ID, event_ID))     
                                
            elif auction.state == "2:Wait_AD_END":
                if auction.auctioneer_ID == self.ID :
                    #Get the IDs of the agents that sent a declaration
                    declaring_agents = self.GetDeclarants(self.auct_decls[auction.ID], auction.ID) 
                    #Check if all the expected participants has sent a declaration
                    #self.print_term("[%s], [%s]" %(declaring_agents,auction.expected_participants))
                    if declaring_agents == auction.expected_participants:
                        #self.print_term(self.identetor("%s [%s] All decls received" %(self.ID, auction.ID)))

                        auction.state = "2:DONE"
                        #the agent evaluates the declarations and define a winner and the expected verifying agents
                        [winner, participants_string, verifiers_string] = self.evaluate_auct_declarations(auction, declaring_agents) 

                        auction_msg = Auction()
                        auction_msg.auction_ID = auction.ID
                        auction_msg.auctioneer_ID = auction.auctioneer_ID
                        auction_msg.state = "AD_END"
                        auction_msg.winner_ID = winner
                        auction_msg.participants_string = participants_string # the expected participants will be updated with 
                                                                              # the winner and the agents that can verify the winner performance
                        auction_msg.verifiers_string = verifiers_string
                        # print(participants_string)
                        # print(verifiers_string)
                        self.auction_pub.publish(auction_msg)
                        

                    #Check if the timeout has been reached
                    elif auction.auct_decl_timeout <= rospy.Time.now() and auction.auct_decl_timeout != rospy.Time(0.0) :
                        auction.state = "2:DONE"
                        #Print/Log
                        missing_declarants = auction.expected_participants.split(", ")
                        for x in declaring_agents.split(", "):
                            for y in auction.expected_participants.split(", "): 
                                if x == y:
                                    missing_declarants.remove(x)
                        missing_decls_string = ""
                        for ID in missing_declarants:
                            missing_decls_string = missing_decls_string + ID + ", "
                        self.print_term(self.identetor("%s [%s] TIMEOUT. Missing auct. decls from [%s] " 
                                                        %(self.ID, auction.ID, missing_decls_string.strip(", "))))
                        self.SAFE_log( "> [%s] Timeout. Missing auct. declarations from [%s]" %(auction.ID, missing_decls_string.strip(", ")) )

                        #If yes, the agent evaluates the received declaration and define a winner and the expected verifying agents
                        [winner, participants_string, verifiers_string] = self.evaluate_auct_declarations(auction, declaring_agents)

                        auction_msg = Auction()
                        auction_msg.auction_ID = auction.ID
                        auction_msg.auctioneer_ID = auction.auctioneer_ID 
                        auction_msg.state = "AD_END"
                        auction_msg.winner_ID = winner
                        auction_msg.participants_string = participants_string # the expected participants will be updated with 
                                                                              # the winner and the agents that can verify the winner performance
                        auction_msg.verifiers_string = verifiers_string

                        # print(participants_string)
                        # print(verifiers_string)
                        self.auction_pub.publish(auction_msg)

                        
                else:
                    #The agent waits...
                    pass

            elif auction.state == "3:AD_END":
                next_state = "4:Wait_Act_START"
                self.StoreNextStateData(auction.ID, next_state)

                agent_action = Agent_Action()
                agent_action.auction_ID = auction.ID
                agent_action.declarant_ID = self.ID
                agent_action.performer_ID = auction.winner
                agent_action.state = "READY_for_Act"
                self.action_pub.publish(agent_action) 

            elif auction.state == "4:Wait_Act_START":
                #self.print_term(self.ID + " VER:" + auction.verifying_agents + " W:" + auction.winner)
                if self.ID == auction.winner:
                    declaring_agents = self.GetDeclarants(self.action_decls[auction.ID], auction.ID) 
                    #Check if all the expected participants has sent a declaration
                    if declaring_agents == auction.expected_participants:
                        #The agents prepares msg with action data
                        agent_action = Agent_Action()
                        agent_action.auction_ID = auction.ID
                        agent_action.performer_ID = self.ID
                        agent_action.state = "Act_START"
                        #The agent sends data to the action result server to simulate the action
                        self.res_server_pub.publish(agent_action)

                        auction.state = "4:DONE"
                        auction_data = auction.ID.split("/") 
                        #The winner performs the action        
                        agent_succ_rate, agent_ver_rates = self.GetRates(auction_data[1], self.ID)
                        if agent_succ_rate != 0.0 :
                            #self.print_term(self.identetor("%s performing %s [%0.3f]" %( self.ID, auction_data[1], agent_succ_rate)))
                            self.SAFE_log("> Everyone is ready!\n > I am performing action [%s]. Estimated success rate = [ %0.3f] \n" 
                                             %(auction_data[1], agent_succ_rate))
                            self.action_pub.publish(agent_action) 
                        else:
                            print("%s>>>>>>>>>>>>>>>ERROR: I WON THE AUCTION, BUT I CAN NOT DO THE ACTION" %self.ID) 
                    
                    elif auction.act_decl_timeout <= rospy.Time.now() and auction.act_decl_timeout != rospy.Time(0.0) :
                        #The agents prepares msg with action data
                        agent_action = Agent_Action()
                        agent_action.auction_ID = auction.ID
                        agent_action.performer_ID = self.ID
                        agent_action.state = "Act_START"
                        #The agent sends data to the action result server to simulate the action
                        self.res_server_pub.publish(agent_action)
                        #Print/Log  
                        missing_declarants = auction.expected_participants.split(", ")
                        for x in declaring_agents.split(", "):
                            for y in auction.expected_participants.split(", "): 
                                if x == y:
                                    missing_declarants.remove(x)
                        missing_decls_string = ""
                        for ID in missing_declarants:
                            missing_decls_string = missing_decls_string + ID + ", "
                        self.print_term(self.identetor("%s [%s] TIMEOUT. Missing act. decls from [%s] " 
                                                        %(self.ID, auction.ID, missing_decls_string.strip(", "))))
                        self.SAFE_log("> [%s] Timeout. Missing act. declarations from [%s]" 
                                        %(auction.ID, missing_decls_string.strip(", ")) )
                
                        auction.state = "4:DONE"
                        auction_data = auction.ID.split("/") 
                        #The winner performs the action        
                        agent_succ_rate, agent_ver_rates = self.GetRates(auction_data[1], self.ID)
                        if agent_succ_rate != 0.0 :
                            self.print_term(self.identetor("%s Performing [%s] Estimated success rate = [ %0.3f]" 
                                                            %(self.ID, auction_data[1], agent_succ_rate)))
                            self.SAFE_log( "> I am performing action [%s]. Estimated success rate = [ %0.3f] \n" 
                                            %(auction_data[1], agent_succ_rate))  
                            self.action_pub.publish(agent_action) 
                        else:
                            print("%s>>>>>>>>>>>>>>>ERROR: I WON THE AUCTION, BUT I CAN NOT DO THE ACTION" %self.ID)
                       
                
                else:
                    #The agent waits...
                    pass

            elif auction.state == "5:Act_PERFORM":
                auction.state = "5:DONE"
                # Send action data to adapter
                if self.adapter_IP != None:
                    data = self.ID + "|PERFORMING|" + auction.ID
                    result_data = self.sendToAdapter(data)
                    auction.result = result_data

                agent_action = Agent_Action()
                agent_action.auction_ID = auction.ID
                agent_action.performer_ID = self.ID
                agent_action.state = "Act_END"
                self.action_pub.publish(agent_action) 
                #self.available = True

            elif auction.state == "5:Act_VERIFICATION":
                if self.adapter_IP != None:
                    data = self.ID + "|VERIFYING|" + auction.ID
                    result_data = self.sendToAdapter(data)
                    auction.result = result_data

                auction.state = "5:DONE"

            elif auction.state == "5:Wait_Act_END":
                pass

            elif auction.state == "6:Act_END":               
                #When an action is finished, check if the agent should verify the action
                # if yes, the agent initialize a Result Declaration in order to declare its perceived result to the other agents
                #the agent requests the outcome of its self-verification on the result of the action
                next_state = "7:Wait_RD"
                self.StoreNextStateData(auction.ID, next_state)  

                auction_data = auction.ID.split("/")
                agent_succ_rate, agent_ver_rates = self.GetRates(auction_data[1], self.ID)
                if auction.verifier:
                    res_decl = Result_Declaration()
                    res_decl.auction_ID = auction.ID
                    res_decl.declarant_ID = self.ID
                    res_decl.performer_ID = auction.winner
                    try:
                        if self.adapter_IP != None:
                            perceived_outcome = auction.result
                        else:
                            #print(self.ID + " WAITING FOR " + auction.ID + " RS")
                            rospy.wait_for_service('action_result', rospy.Duration(self.timeout))
                            #print(self.ID + " RESULT READY")
                            resp = self.action_result(auction.ID, auction.winner, self.ID)
                            perceived_outcome = resp.perceived_result

                        res_decl.perceived_outcome = perceived_outcome
                                
                        #self.print_term(self.identetor("%s verifying %s(%s) -> %s [%0.3f, %0.3f]"
                        #%(self.ID, auction_data[1], auction.winner, res_decl.perceived_outcome, agent_ver_rates[0], agent_ver_rates[1])))
                        self.SAFE_log(("> Action [%s] is finished, result perceived as [%s]." %(auction_data[1], res_decl.perceived_outcome) + 
                                        "Estimated verification rates = [ %0.3f, %0.3f]" %(agent_ver_rates[0], agent_ver_rates[1])))
                        
                        self.ver_decl_pub.publish(res_decl)
                        
                    except rospy.ServiceException as e:
                        self.print_term(self.identetor("%s Service call during [%s] failed: %s"%(self.ID, auction.ID, e)))

                else:
                    self.SAFE_log("> Action [%s] is finished, but I cannot verify the result " %( auction_data[1])) 
                        
            elif auction.state == "7:Wait_RD":
                #Get the IDs of the agents that sent a declaration
                declaring_agents = self.GetDeclarants(self.verif_decls[auction.ID], auction.ID)  
                #Check if all the expected participants has sent a declaration...
                if declaring_agents == auction.verifying_agents:
                    #self.print_term(self.identetor("%s [%s] All decls received." %(self.ID, auction.ID)))
                    #If yes, the agent evaluates the received verification declaration and updates its metrics
                    self.evaluate_result_declarations(auction)
                    next_state = "END" 
                    self.StoreNextStateData(auction.ID, next_state)
                #.. Or check if the timeout has been reached
                elif auction.res_decl_timeout <= rospy.Time.now() and auction.res_decl_timeout != rospy.Time(0.0):
                    #Print/Log
                    missing_declarants = auction.verifying_agents.split(", ")
                    for x in declaring_agents.split(", "):
                        for y in auction.verifying_agents.split(", "): 
                            if x == y:
                                missing_declarants.remove(x)
                    missing_decls_string = ""
                    for ID in missing_declarants:
                        missing_decls_string = missing_decls_string + ID + ", "
                    self.print_term(self.identetor("%s [%s] TIMEOUT. Missing res. decls from [%s] "
                                                  %(self.ID, auction.ID, missing_decls_string.strip(", "))))
                    self.SAFE_log( "> [%s] Timeout. Missing res. declarations from [%s]" %(auction.ID, missing_decls_string.strip(", ")))
                
                    #The agent evaluates the received verification declaration and updates its metrics                
                    self.evaluate_result_declarations(auction)
                    next_state = "END"
                    self.StoreNextStateData(auction.ID, next_state)
                    
            elif auction.state in ["1:DONE", "2:DONE", "3:DONE", "4:DONE", "5:DONE", "6:DONE", "7:DONE", "END"]:
                #After completing the task relative to a given state, the agent waits, ready for updates of auction state
                pass

            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Something strange happend with auction %s state: %s " 
                                %(self.ID, auction.ID, auction.state) )            

    def AuctionStarter(self):
            for auction in self.auctions_queue[:]:
                auction_data = auction.split()
                event_ID = auction_data[0]
                event_counter = auction_data[1]
                auctioneer_ID = auction_data[2]

            
                if len(event_ID.split(":"))> 1: 
                    if event_ID.split(":")[1] == "SERIAL" and event_ID in self.plans.keys():
                        temp = []
                        for action_ID in self.plans[event_ID]: 
                            auction_ID = event_ID + "/" + action_ID + "/" + str(event_counter)
                            self.print_term(self.identetor("%s Init [%s]" %(self.ID,auction_ID)))
                            self.SAFE_log( ("\n" + "> Initializing auction [%s]" %auction_ID ))
                            new_auction_state = "1:AD_START"
                            new_auction_data = [auction_ID, auctioneer_ID]
                            temp.append([auction_ID, new_auction_state, new_auction_data])

                        first_serial_action_data = temp.pop(0)
                        self.startNextSerialAuction(first_serial_action_data[0], first_serial_action_data[1], first_serial_action_data[2])
                        self.serial_auctions[event_ID + "/" + event_counter] = temp     
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. The event or the event type is unknown" %self.ID)

                elif event_ID in self.plans.keys(): 
                    for action_ID in self.plans[event_ID]: 
                        auction_ID = event_ID + "/" + action_ID + "/" + str(event_counter)
                        self.print_term(self.identetor("%s Init [%s]" %(self.ID,auction_ID)))
                        self.SAFE_log( ("\n" + "> Initializing auction [%s]" %auction_ID ))
                        new_auction_state = "1:AD_START"
                        new_auction_data = [auction_ID, auctioneer_ID]
                        self.StoreNextStateData(auction_ID, new_auction_state, new_auction_data)
                        self.auct_decls[auction_ID] = []   
                        self.verif_decls[auction_ID] = []
                        self.action_decls[auction_ID] = []

                        if self.adapter_IP != None:
                            data = self.ID + "|DECLARING_IMMEDIATE|" + "Hey, boys. We need someone to say " + action_ID
                            self.sendToAdapter(data)

                        auction_msg = Auction()
                        auction_msg.auction_ID = auction_ID
                        auction_msg.auctioneer_ID = self.ID
                        auction_msg.state = "AD_START"
                        self.auction_pub.publish(auction_msg)  
                            
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. The event is unknown" %self.ID)

                self.auctions_queue.remove(auction)    
   
    def evaluate_auct_declarations(self, auction, declaring_agents): #Method to evaluate the received Auction Declarations in order 
                                                                     #to choose the winner
        winner_perc_comp = 0.0 
        winner_ID = ""
        verifiers_string = "" 
        participants_string = ""
        declaring_agents_list = declaring_agents.split(", ")
        verifiers_list = []
        participants_list = []
        perc_comp_leaderboard = []
        auction_ID = auction.ID
        auction_data = auction_ID.split("/")
        event_ID = auction_data[0]
        action_ID = auction_data[1]
        self.SAFE_log(self.log_auction_declarations(auction))
        
        ## BCIC MODE ##
        if self.mode == "BCIC":
            perc_comp_leaderboard, winner_perc_comp = self.evaluate_BCIC(auction_ID, action_ID, declaring_agents_list)
        
        ## WINDOW MODE ##
        elif self.mode == "WINDOW":   
            perc_comp_leaderboard, winner_perc_comp = self.evaluate_WINDOW(auction_ID, action_ID, declaring_agents_list)

        ## BOOT MODE ##
        elif self.mode == "BOOT":
            perc_comp_leaderboard, winner_perc_comp = self.evaluate_BOOT(auction_ID, action_ID, declaring_agents_list)

        ## TEST MODE ##
        elif self.mode == "TEST":
            perc_comp_leaderboard, winner_perc_comp = self.evaluate_TEST(auction_ID, declaring_agents_list)
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. MODE NOT RECOGNIZED" %(self.ID))   
        

        #CHOOSE WINNER 
        #Choosing the winner randomly between the agents with the best perceived competence
        if len(perc_comp_leaderboard) == 0:
            self.print_term(self.ID + ">>>>>>>>>>>>>>>ERROR. NO WINNER FOR THIS AUCTION")

        elif len(perc_comp_leaderboard) == 1:
            winner_ID = perc_comp_leaderboard.pop(0)
        else:
            winner_index = random.randint(0,len(perc_comp_leaderboard)-1)
            winner_ID = perc_comp_leaderboard.pop(winner_index)
            #print("CHECK")

        self.SAFE_log( "> Winner-estimated success rate: [%0.3f]" %winner_perc_comp)

        if self.adapter_IP != None:
            decl_data = "I declare that " + winner_ID + " will be given the responsibility of saying " + action_ID
            data = self.ID + "|DECLARING_IMMEDIATE|" + decl_data
            self.sendToAdapter(data)

        #DEFINE EXPECTED VERIFIERS
        #Check which agents, between the ones that sent a declaration, are able to verify the winner      
        for decl in self.auct_decls[auction_ID]:
            for rate in decl.rates_list:
                if rate.verification_rates[0] > 0.0 and rate.verification_rates[1] > 0.0 and decl.declarant_ID not in verifiers_list:
                    verifiers_list.append(decl.declarant_ID)
        for agent_ID in sorted(verifiers_list):
            verifiers_string +=  agent_ID + ", "          
        verifiers_string = verifiers_string.strip(", ")   

        #DEFINE EXPECTED PARTICIPANTS
        participants_list = verifiers_list[:]
        if winner_ID not in participants_list:
            participants_list.append(winner_ID)

        for agent_ID in sorted(participants_list):
            participants_string += agent_ID + ", "          
        participants_string = participants_string.strip(", ")     

        self.print_term(self.identetor("%s [%s] DA:" %(self.ID, auction_ID) + "%s" %(declaring_agents)))
        self.print_term(self.identetor("%s [%s] VA:" %(self.ID, auction_ID) + "%s" %(verifiers_string))) 
        self.SAFE_log("> I have received declarations from: %s\n" %(declaring_agents))
        
        return winner_ID, participants_string, verifiers_string

    def evaluate_BCIC(self, auction_ID, action_ID, declaring_agents_list):
        perc_comp_dict = {}
        perc_comp_leaderboard = []
        winner_perc_comp = 0.0
        log_string = ""
        w_rels_string = ""
        
        if self.behavior == "Individualism":
            #FOR EACH DECLARING AGENT (1 DECLARING AGENT FOR EACH DECLARATION) THE AGENT EVALUATES ITS PERCEIVED COMPETENCE
            for decl in self.auct_decls[auction_ID]:
                candidate_perf_ID = decl.declarant_ID
                candidate_decl_reliability = 0.0
                est_reliability = 0.0

                candidate_decl_reliability = self.GetDeclarantReliability(candidate_perf_ID, decl.rates_list)
            
                perc_comp_dict.update(self.IndividualismBehavior_BCIC(action_ID, candidate_perf_ID, candidate_decl_reliability))

            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard:
                    perc_comp_leaderboard.append(performer_ID)

        elif self.behavior == "Collectivism":

            candidates_reliability_dict = {}
            declaring_agents_vtw = {}

            #Ierating through the declarations to store the reliability data 
            for decl in self.auct_decls[auction_ID]:
                if decl.declarant_ID in self.trust_metric_db.keys():
                    #Check if the agent has any experience about the candidate 
                    if action_ID in self.trust_metric_db[decl.declarant_ID].keys():
                        vtw_key = action_ID + ", " + self.ID + "," + decl.declarant_ID
                        declaring_agents_vtw[vtw_key] = self.trust_metric_db[decl.declarant_ID][action_ID].ver_trustworthiness
                for rate in decl.rates_list:  
                    if rate.performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration  
                    if self.disposition == "Optimist":
                        #Upper bound of BCI
                        est_reliability = rate.upper_sr
                    elif self.disposition == "Pessimist":
                        #Lower bound of BCI
                        est_reliability = rate.lower_sr
                    elif self.disposition == "Realist":
                        est_reliability = rate.success_rate
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
                    #Reliability in action "a", according to the declarant "k", of performer "j"
                    candidates_reliability_dict[action_ID +", "+ decl.declarant_ID + ", " + rate.performer_ID] = est_reliability
            log_string += ("DECLARING AGENTS VTW: " + str(declaring_agents_vtw) + "\n")
            log_string += ("RELIABILITIES: " + str(candidates_reliability_dict) + "\n")
            
            #PER OGNI DICHIARANTE CALCOLO LA W_REL ASSOCIATA A ESSO, SE POSSIBILE
            for candidate_perf_ID in declaring_agents_list:
                w_rels= []
                vtw_weights = []
                w_rels_string += "WEIGHTED RELIABILITY FOR %s = " %candidate_perf_ID
                for rel_key in sorted(candidates_reliability_dict.keys()):
                    decl_agent_ID = rel_key.split(", ")[1]
                    performer_ID = rel_key.split(", ")[2]
                    vtw_key = action_ID + ", " + self.ID + "," + decl_agent_ID
                    if candidate_perf_ID == performer_ID and vtw_key in declaring_agents_vtw.keys():
                        if declaring_agents_vtw[vtw_key] > 0:
                            vtw = declaring_agents_vtw[action_ID + ", " + self.ID + "," + decl_agent_ID]
                            reliability = candidates_reliability_dict[rel_key]
                            w_rels.append(vtw * reliability)
                            vtw_weights.append(vtw)
                            w_rels_string += (" %0.3f * %0.3f +"  %(vtw, reliability))
                w_rels_string = w_rels_string.strip("+") + "\n"

                candidate_decl_reliability = candidates_reliability_dict[action_ID +", "+ candidate_perf_ID + ", " + candidate_perf_ID]

                perc_comp_dict.update(self.CollectivismBehavior_BCIC(candidate_perf_ID, candidate_decl_reliability, 
                                                                     w_rels, vtw_weights))

            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard:
                    perc_comp_leaderboard.append(performer_ID)
                
        else: 
            print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my behaviour" %(self.ID))
        
        print(log_string)
        self.SAFE_log(log_string)
        print(w_rels_string)
        self.SAFE_log(w_rels_string)
        print(perc_comp_dict)
        self.SAFE_log("PERCEIVED COMPETENCE: " + str(perc_comp_dict))

        if self.adapter_IP != None:
            decl_data = "According to me, "
            for agent_ID, perc_comp in perc_comp_dict.items():
                if perc_comp <= 0.50:
                    decl_data += agent_ID + " is not trustable at all when saying " + action_ID + ". "
                elif perc_comp > 0.50 and perc_comp < 0.70:
                    decl_data += agent_ID + " is bad at saying" + action_ID + ". "
                elif perc_comp >= 0.70 and perc_comp <= 0.85:
                    decl_data += agent_ID + " is good at saying " + action_ID + ". "
                elif perc_comp > 0.85:
                    decl_data += agent_ID + " is great at saying " + action_ID + ". "

            data = self.ID + "|DECLARING_IMMEDIATE|" + decl_data.strip(". ")
            self.sendToAdapter(data)

        print(perc_comp_leaderboard)
        return perc_comp_leaderboard, winner_perc_comp

    def IndividualismBehavior_BCIC(self, action_ID, candidate_perf_ID, candidate_decl_reliability):
        perc_comp_dict = dict()
        #Check if the agent has prior experiences with the declaring agent
        if candidate_perf_ID in self.trust_metric_db.keys():
            #Check if the agent has prior experiences with the declaring agent performing the action
            if action_ID in self.trust_metric_db[candidate_perf_ID].keys():
                tr_mtr = self.trust_metric_db[candidate_perf_ID][action_ID]
                if len(tr_mtr.results) > 0:
                    print(candidate_perf_ID + ": CASE_1")
                    obs_rel, lower_p, upper_p = self.compute_BCI_bounds(tr_mtr)
            
                    if self.disposition == "Optimist":
                        #Upper bound of BCI
                        est_reliability = upper_p
                    elif self.disposition == "Pessimist":
                        #Lower bound of BCI
                        est_reliability = lower_p
                    elif self.disposition == "Realist":
                        #Strictly observed reliability
                        est_reliability = obs_rel
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
                else:
                    print(candidate_perf_ID + ": CASE_01")
                    est_reliability = candidate_decl_reliability
            
                perc_comp_dict[candidate_perf_ID] = est_reliability

            #Check if the agent has prior experiences with the declaring agent performing other actions
            elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:
                #If not, the agent, according to its disposition, evaluates
                max_reliability = -1.0
                min_reliability = 2.0
                summ_n_act = 0.0
                summ_reliability = 0.0
                
                for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                    obs_rel, lower_p, upper_p = self.compute_BCI_bounds(tr_mtr, no_warning=False)

                    if self.disposition == "Optimist":
                        if len(tr_mtr.results) > 0 and upper_p >= max_reliability:
                                max_reliability = upper_p
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    elif self.disposition == "Pessimist":
                        if len(tr_mtr.results) > 0 and lower_p <= min_reliability:
                                min_reliability = lower_p
                    elif self.disposition == "Realist":
                        if len(tr_mtr.results) > 0:
                            summ_reliability += len(tr_mtr.results) * tr_mtr.reliability
                            summ_n_act += len(tr_mtr.results)
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

                if self.disposition == "Optimist":
                    if max_reliability != -1.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Pessimist":
                    if min_reliability != 2.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = min_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Realist":
                    if summ_reliability != 0.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                    
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
            else:
                print(candidate_perf_ID + ": CASE_0")
                if self.disposition == "Optimist":
                    #Upper bound of BCI
                    perc_comp_dict[candidate_perf_ID] = 1.0
                elif self.disposition == "Pessimist":
                    #Lower bound of BCI
                    perc_comp_dict[candidate_perf_ID] = 0.0
                elif self.disposition == "Realist":
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with this agent. I don't know him" %(self.ID))
        
        return perc_comp_dict

    def CollectivismBehavior_BCIC(self, candidate_perf_ID, candidate_decl_reliability, w_rels, vtw_weights):
        perc_comp_dict = dict()

        if len(w_rels) != 0:
            print(candidate_perf_ID + ": CASE_1")
            summ_w_rel = 0.0
            summ_vtw = 0.0
            for w_rel in w_rels:
                summ_w_rel += w_rel
            for vtw_weight in vtw_weights:
                summ_vtw += vtw_weight
            perc_comp_dict[candidate_perf_ID] = summ_w_rel/summ_vtw
        #Check if the agent has prior experiences with the declaring agent performing other actions
        elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:
            #If not, the agent, according to its disposition, evaluates the reliability based on other actions reliability
            max_reliability = -1.0
            min_reliability = 2.0
            summ_n_act = 0.0
            summ_reliability = 0.0

            for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                obs_rel, lower_p, upper_p = self.compute_BCI_bounds(tr_mtr, no_warning=False)

                if self.disposition == "Optimist":
                    w_rel = tr_mtr.ver_trustworthiness * upper_p
                    if len(tr_mtr.results) > 0 and w_rel >= max_reliability:
                        max_reliability = w_rel
                elif self.disposition == "Pessimist":
                    w_rel = tr_mtr.ver_trustworthiness * lower_p
                    if len(tr_mtr.results) > 0 and w_rel <= min_reliability:
                        min_reliability = w_rel
                elif self.disposition == "Realist": 
                    if len(tr_mtr.results) > 0:
                        summ_reliability += len(tr_mtr.results) * tr_mtr.ver_trustworthiness * obs_rel
                        summ_n_act += len(tr_mtr.results)
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
                

            if self.disposition == "Optimist":
                if max_reliability != -1.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = max_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Pessimist":
                if min_reliability != 2.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = min_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Realist":
                if summ_reliability != 0.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
        
        else:
            if self.disposition == "Optimist":
                    #Upper bound of BCI
                    perc_comp_dict[candidate_perf_ID] = 1.0
            elif self.disposition == "Pessimist":
                    #Lower bound of BCI
                    perc_comp_dict[candidate_perf_ID] = 0.0
            elif self.disposition == "Realist":
                perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
        
        return perc_comp_dict

    def evaluate_WINDOW(self, auction_ID, action_ID, declaring_agents_list):
        perc_comp_dict = {}
        perc_comp_leaderboard = []
        winner_perc_comp = 0.0 

        if self.behavior == "Individualism":

            for decl in self.auct_decls[auction_ID]:
                candidate_perf_ID = decl.declarant_ID
                candidate_decl_reliability = self.GetDeclarantReliability(candidate_perf_ID, decl.rates_list)
                #Check if the agent has prior experiences with the declaring agent
                perc_comp_dict.update(self.IndividualismBehavior_WINDOW(action_ID, candidate_perf_ID, candidate_decl_reliability))

            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard:
                    perc_comp_leaderboard.append(performer_ID)

        elif self.behavior == "Collectivism":

            candidates_reliability_dict = {}
            declaring_agents_vtw = {}

            #Ierating through the declarations to store the reliability data 
            for decl in self.auct_decls[auction_ID]:
                if decl.declarant_ID in self.trust_metric_db.keys():
                    #Check if the agent has any experience about the candidate 
                    if action_ID in self.trust_metric_db[decl.declarant_ID].keys():
                        vtw_key = action_ID + ", " + self.ID + "," + decl.declarant_ID
                        declaring_agents_vtw[vtw_key] = self.trust_metric_db[decl.declarant_ID][action_ID].ver_trustworthiness
                for rate in decl.rates_list:  
                    if rate.performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration  
                    #Reliability in action "a", according to the declarant "k", of performer "j"
                    candidates_reliability_dict[action_ID +", "+ decl.declarant_ID + ", " + rate.performer_ID] = rate.success_rate
            
            print("DECLARING AGENTS VTW: " + str(declaring_agents_vtw) + "\n" + "RELIABILITIES: " + str(candidates_reliability_dict))
            self.SAFE_log("DECLARING AGENTS VTW: " + str(declaring_agents_vtw) + "\n" + "RELIABILITIES: " + str(candidates_reliability_dict))
            for candidate_perf_ID in declaring_agents_list:
                w_rels= []
                vtw_weights = []
                w_rels_string = "WEIGHTED RELIABILITY FOR %s = " %candidate_perf_ID
                for rel_key in sorted(candidates_reliability_dict.keys()):
                    decl_agent_ID = rel_key.split(", ")[1]
                    performer_ID = rel_key.split(", ")[2]
                    vtw_key = action_ID + ", " + self.ID + "," + decl_agent_ID
                    if candidate_perf_ID == performer_ID and vtw_key in declaring_agents_vtw.keys():
                        if declaring_agents_vtw[vtw_key] > 0:
                            vtw = declaring_agents_vtw[action_ID + ", " + self.ID + "," + decl_agent_ID]
                            reliability = candidates_reliability_dict[rel_key]
                            w_rels.append( vtw * reliability)
                            vtw_weights.append(vtw)
                            w_rels_string += (" %0.3f * %0.3f +"  %(vtw, reliability))
                print(w_rels_string.strip("+"))
                candidate_decl_reliability = candidates_reliability_dict[action_ID +", "+ candidate_perf_ID + ", " + candidate_perf_ID]

                perc_comp_dict.update(self.CollectivismBehavior_WINDOW(candidate_perf_ID, candidate_decl_reliability, 
                                                                       w_rels, vtw_weights))
           
            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard: 
                    perc_comp_leaderboard.append(performer_ID)
                
        else: #BLIND COLLECTIVISM: choose the best performer according the unweighted opinion of everyone
            for decl in self.auct_decls[auction_ID]:
                for rate in decl.rates_list:
                    if rate.success_rate > winner_perc_comp:      
                        if rate.performer_ID not in declaring_agents_list:
                            continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                        winner_perc_comp = rate.success_rate
                        perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                    if rate.success_rate == winner_perc_comp and rate.performer_ID not in perc_comp_leaderboard :
                        perc_comp_leaderboard.append(rate.performer_ID)
                        
        print("PERCEIVED COMPETENCE: " + str(perc_comp_dict) +"\n" + "LEADERBOARD" + str(perc_comp_leaderboard))
        self.SAFE_log("PERCEIVED COMPETENCE: " + str(perc_comp_dict) +"\n" + "LEADERBOARD" + str(perc_comp_leaderboard))    
        return perc_comp_leaderboard, winner_perc_comp
      
    def IndividualismBehavior_WINDOW(self,action_ID, candidate_perf_ID, candidate_decl_reliability):
        perc_comp_dict = dict()

        if candidate_perf_ID in self.trust_metric_db.keys():
            #Check if the agent has prior experiences with the declaring agent performing the action
            if action_ID in self.trust_metric_db[candidate_perf_ID].keys():
                tr_mtr = self.trust_metric_db[candidate_perf_ID][action_ID]
                if len(tr_mtr.results) >= WINDOW_LENGTH:
                    print(candidate_perf_ID + ": CASE_1")
                    perc_comp_dict[candidate_perf_ID] = tr_mtr.reliability
                else:
                    print(candidate_perf_ID + ": CASE_01")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                

            #Check if the agent has prior experiences with the declaring agent performing other actions
            elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:
                print(self.behavior + " " + self.disposition +" on " + candidate_perf_ID)
                summ_n_act = 0.0
                summ_reliability = 0.0
                max_reliability = -1.0
                min_reliability = 2.0
                for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                    if self.disposition == "Optimist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH and tr_mtr.reliability >= max_reliability:
                            max_reliability = tr_mtr.reliability
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    elif self.disposition == "Pessimist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH and tr_mtr.reliability <= min_reliability:
                            min_reliability = tr_mtr.reliability
                    elif self.disposition == "Realist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH:
                            summ_reliability += len(tr_mtr.results) * tr_mtr.reliability
                            summ_n_act += len(tr_mtr.results)
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

                if self.disposition == "Optimist":
                    if max_reliability != -1.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Pessimist":
                    if min_reliability != 2.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = min_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Realist":
                    if summ_reliability != 0.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                    
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

            else:
                print(candidate_perf_ID + ": CASE_0")
                perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with this agent. I don't know him" %(self.ID))

        return perc_comp_dict

    def CollectivismBehavior_WINDOW(self, candidate_perf_ID, candidate_decl_reliability, w_rels, vtw_weights):
        perc_comp_dict = dict()
        #The agent check if it is possible to compute a weighted reliability for the candidate
        if len(w_rels) != 0:
            print(candidate_perf_ID + ": CASE_1")
            summ_w_rel = 0.0
            summ_vtw = 0.0
            for w_rel in w_rels:
                summ_w_rel += w_rel
            for vtw_weight in vtw_weights:
                summ_vtw += vtw_weight
            perc_comp_dict[candidate_perf_ID] = summ_w_rel/summ_vtw
            
        #If not, the agent, according to its disposition, evaluates the reliability based on other actions reliability
        elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:

            print(self.behavior + " " + self.disposition +" on " + candidate_perf_ID)

            min_reliability = 2.0
            max_reliability = -1.0
            summ_n_act = 0.0
            summ_reliability = 0.0

            for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                if self.disposition == "Optimist":
                    w_rel = tr_mtr.ver_trustworthiness * tr_mtr.reliability
                    if len(tr_mtr.results) >= WINDOW_LENGTH and w_rel >= max_reliability:
                        max_reliability = w_rel
                elif self.disposition == "Pessimist":
                    w_rel = tr_mtr.ver_trustworthiness * tr_mtr.reliability
                    if len(tr_mtr.results) >= WINDOW_LENGTH and w_rel <= min_reliability:
                        min_reliability = w_rel
                elif self.disposition == "Realist": 
                    if len(tr_mtr.results) >= WINDOW_LENGTH:
                        summ_reliability += len(tr_mtr.results) * tr_mtr.ver_trustworthiness * tr_mtr.reliability
                        summ_n_act += len(tr_mtr.results)
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

            if self.disposition == "Optimist":
                if max_reliability != -1.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = max_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Pessimist":
                if min_reliability != 2.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = min_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Realist":
                if summ_reliability != 0.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
        else:
            print(candidate_perf_ID + ": CASE_0")
            perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability 

        return perc_comp_dict
      
    def evaluate_BOOT(self, auction_ID, action_ID, declaring_agents_list):
        perc_comp_dict = {}
        perc_comp_leaderboard = []
        winner_perc_comp = 0.0 

        if self.behavior == "Individualism":

            for decl in self.auct_decls[auction_ID]:
                candidate_perf_ID = decl.declarant_ID
                candidate_decl_reliability = self.GetDeclarantReliability(candidate_perf_ID, decl.rates_list)
                #Check if the agent has prior experiences with the declaring agent
                perc_comp_dict.update(self.IndividualismBehavior_BOOT(action_ID, candidate_perf_ID, candidate_decl_reliability))

            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard:
                    perc_comp_leaderboard.append(performer_ID)

        elif self.behavior == "Collectivism":

            candidates_reliability_dict = {}
            declaring_agents_vtw = {}

            #Ierating through the declarations to store the reliability data 
            for decl in self.auct_decls[auction_ID]:
                if decl.declarant_ID in self.trust_metric_db.keys():
                    #Check if the agent has any experience about the candidate 
                    if action_ID in self.trust_metric_db[decl.declarant_ID].keys():
                        vtw_key = action_ID + ", " + self.ID + "," + decl.declarant_ID
                        declaring_agents_vtw[vtw_key] = self.trust_metric_db[decl.declarant_ID][action_ID].ver_trustworthiness
                for rate in decl.rates_list:  
                    if rate.performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration  
                    #Reliability in action "a", according to the declarant "k", of performer "j"
                    candidates_reliability_dict[action_ID +", "+ decl.declarant_ID + ", " + rate.performer_ID] = rate.success_rate
            
            print("DECLARING AGENTS VTW: " + str(declaring_agents_vtw) + "\n" + "RELIABILITIES: " + str(candidates_reliability_dict))
            self.SAFE_log("DECLARING AGENTS VTW: " + str(declaring_agents_vtw) + "\n" + "RELIABILITIES: " + str(candidates_reliability_dict))
            for candidate_perf_ID in declaring_agents_list:
                w_rels= []
                vtw_weights = []
                w_rels_string = "WEIGHTED RELIABILITY FOR %s = " %candidate_perf_ID
                for rel_key in sorted(candidates_reliability_dict.keys()):
                    decl_agent_ID = rel_key.split(", ")[1]
                    performer_ID = rel_key.split(", ")[2]
                    vtw_key = action_ID + ", " + self.ID + "," + decl_agent_ID
                    if candidate_perf_ID == performer_ID and vtw_key in declaring_agents_vtw.keys():
                        if declaring_agents_vtw[vtw_key] > 0:
                            vtw = declaring_agents_vtw[action_ID + ", " + self.ID + "," + decl_agent_ID]
                            reliability = candidates_reliability_dict[rel_key]
                            w_rels.append( vtw * reliability)
                            vtw_weights.append(vtw)
                            w_rels_string += (" %0.3f * %0.3f +"  %(vtw, reliability))
                print(w_rels_string.strip("+"))
                candidate_decl_reliability = candidates_reliability_dict[action_ID +", "+ candidate_perf_ID + ", " + candidate_perf_ID]

                perc_comp_dict.update(self.CollectivismBehavior_BOOT(candidate_perf_ID, candidate_decl_reliability, 
                                                                       w_rels, vtw_weights))
           
            for performer_ID,w_rel in perc_comp_dict.items():
                if w_rel > winner_perc_comp: 
                    if performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = w_rel
                    perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                if w_rel == winner_perc_comp and performer_ID not in perc_comp_leaderboard: 
                    perc_comp_leaderboard.append(performer_ID)
                
        else: #BLIND COLLECTIVISM: choose the best performer according the unweighted opinion of everyone
            for decl in self.auct_decls[auction_ID]:
                for rate in decl.rates_list:
                    if rate.success_rate > winner_perc_comp:      
                        if rate.performer_ID not in declaring_agents_list:
                            continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                        winner_perc_comp = rate.success_rate
                        perc_comp_leaderboard = [] #the winner ID is added immediatly afterward 
                    if rate.success_rate == winner_perc_comp and rate.performer_ID not in perc_comp_leaderboard :
                        perc_comp_leaderboard.append(rate.performer_ID)
                        
        print("PERCEIVED COMPETENCE: " + str(perc_comp_dict) +"\n" + "LEADERBOARD" + str(perc_comp_leaderboard))
        self.SAFE_log("PERCEIVED COMPETENCE: " + str(perc_comp_dict) +"\n" + "LEADERBOARD" + str(perc_comp_leaderboard))    
        return perc_comp_leaderboard, winner_perc_comp

    def IndividualismBehavior_BOOT(self,action_ID, candidate_perf_ID, candidate_decl_reliability):
        perc_comp_dict = dict()

        if candidate_perf_ID in self.trust_metric_db.keys():
            #Check if the agent has prior experiences with the declaring agent performing the action
            if action_ID in self.trust_metric_db[candidate_perf_ID].keys():
                tr_mtr = self.trust_metric_db[candidate_perf_ID][action_ID]
                if len(tr_mtr.results) >= WINDOW_LENGTH:
                    print(candidate_perf_ID + ": CASE_1")
                    perc_comp_dict[candidate_perf_ID] = tr_mtr.reliability
                else:
                    print(candidate_perf_ID + ": CASE_01")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                

            #Check if the agent has prior experiences with the declaring agent performing other actions
            elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:
                print(self.behavior + " " + self.disposition +" on " + candidate_perf_ID)
                summ_n_act = 0.0
                summ_reliability = 0.0
                max_reliability = -1.0
                min_reliability = 2.0
                for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                    if self.disposition == "Optimist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH and tr_mtr.reliability >= max_reliability:
                            max_reliability = tr_mtr.reliability
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    elif self.disposition == "Pessimist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH and tr_mtr.reliability <= min_reliability:
                            min_reliability = tr_mtr.reliability
                    elif self.disposition == "Realist":
                        if len(tr_mtr.results) >= WINDOW_LENGTH:
                            summ_reliability += len(tr_mtr.results) * tr_mtr.reliability
                            summ_n_act += len(tr_mtr.results)
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

                if self.disposition == "Optimist":
                    if max_reliability != -1.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = max_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Pessimist":
                    if min_reliability != 2.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = min_reliability
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                elif self.disposition == "Realist":
                    if summ_reliability != 0.0:
                        print(candidate_perf_ID + ": CASE_2")
                        perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                    else:
                        print(candidate_perf_ID + ": CASE_02")
                        perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
                    
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

            else:
                print(candidate_perf_ID + ": CASE_0")
                perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with this agent. I don't know him" %(self.ID))

        return perc_comp_dict

    def CollectivismBehavior_BOOT(self, candidate_perf_ID, candidate_decl_reliability, w_rels, vtw_weights):
        perc_comp_dict = dict()
        #The agent check if it is possible to compute a weighted reliability for the candidate
        if len(w_rels) != 0:
            print(candidate_perf_ID + ": CASE_1")
            summ_w_rel = 0.0
            summ_vtw = 0.0
            for w_rel in w_rels:
                summ_w_rel += w_rel
            for vtw_weight in vtw_weights:
                summ_vtw += vtw_weight
            perc_comp_dict[candidate_perf_ID] = summ_w_rel/summ_vtw
            
        #If not, the agent, according to its disposition, evaluates the reliability based on other actions reliability
        elif len(self.trust_metric_db[candidate_perf_ID].keys()) != 0:

            print(self.behavior + " " + self.disposition +" on " + candidate_perf_ID)

            min_reliability = 2.0
            max_reliability = -1.0
            summ_n_act = 0.0
            summ_reliability = 0.0

            for tr_mtr in self.trust_metric_db[candidate_perf_ID].values():
                if self.disposition == "Optimist":
                    w_rel = tr_mtr.ver_trustworthiness * tr_mtr.reliability
                    if len(tr_mtr.results) >= WINDOW_LENGTH and w_rel >= max_reliability:
                        max_reliability = w_rel
                elif self.disposition == "Pessimist":
                    w_rel = tr_mtr.ver_trustworthiness * tr_mtr.reliability
                    if len(tr_mtr.results) >= WINDOW_LENGTH and w_rel <= min_reliability:
                        min_reliability = w_rel
                elif self.disposition == "Realist": 
                    if len(tr_mtr.results) >= WINDOW_LENGTH:
                        summ_reliability += len(tr_mtr.results) * tr_mtr.ver_trustworthiness * tr_mtr.reliability
                        summ_n_act += len(tr_mtr.results)
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

            if self.disposition == "Optimist":
                if max_reliability != -1.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = max_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Pessimist":
                if min_reliability != 2.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = min_reliability
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            elif self.disposition == "Realist":
                if summ_reliability != 0.0:
                    print(candidate_perf_ID + ": CASE_2")
                    perc_comp_dict[candidate_perf_ID] = summ_reliability/summ_n_act
                else:
                    print(candidate_perf_ID + ": CASE_02")
                    perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))
        else:
            print(candidate_perf_ID + ": CASE_0")
            perc_comp_dict[candidate_perf_ID] = candidate_decl_reliability 

        return perc_comp_dict

    def evaluate_TEST(self,auction_ID, declaring_agents_list):
        perc_comp_leaderboard = []
        winner_perc_comp = 0.0 

        for decl in self.auct_decls[auction_ID]:
            for rate in decl.rates_list:
                if rate.success_rate > winner_perc_comp:      
                    if rate.performer_ID not in declaring_agents_list:
                        continue   # To prevent that an agent that did not declared is declared the winner due to other agents declaration                  
                    winner_perc_comp = rate.success_rate
                    perc_comp_leaderboard = []
                if rate.success_rate == winner_perc_comp and rate.performer_ID not in perc_comp_leaderboard:
                    perc_comp_leaderboard.append(rate.performer_ID)

        return perc_comp_leaderboard, winner_perc_comp
        
    def GetDeclarantReliability(self, agent_ID, rates_list):
        declared_reliability = 0.0
        for rate in rates_list:
            if agent_ID == rate.performer_ID: # Check only for the reliability of the declaring agent
                declared_reliability = rate.success_rate
                break
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Declarant has not declared its own reliability" %(self.ID))
        
        return declared_reliability

    def evaluate_result_declarations(self, auction): #Method for trust metrics update

        self.SAFE_log(self.log_result_declarations(auction))
        self.update_trust_metrics(auction)   
        
        self.verif_decls.pop(auction.ID)
        self.auct_decls.pop(auction.ID)
        self.action_decls.pop(auction.ID)                                            
            
    def update_trust_metrics(self,auction):
        #Create or get the tr_mtr metric related to the performer
        
        positive_voters = []
        negative_voters = []
        negative_voters_string = ""
        positive_voters_string = ""

        #Check to avoid verify if the agent declaration and the performer declaration are coincident (when the agent is the performer)
        if auction.winner == self.ID:
            my_decl = self.GetResDeclaration(auction.ID, self.ID) 
            performer_decl = my_decl
        else:
            if auction.verifier:
                my_decl = self.GetResDeclaration(auction.ID, self.ID) 
                performer_decl = self.GetResDeclaration(auction.ID, auction.winner)
            else:
                my_decl = None
                performer_decl = self.GetResDeclaration(auction.ID, auction.winner)

        for decl in self.verif_decls[auction.ID]:
            if decl.perceived_outcome == "True":
                positive_voters.append(decl.declarant_ID)
                positive_voters_string = positive_voters_string + decl.declarant_ID + ", "
            if decl.perceived_outcome == "False":
                negative_voters.append(decl.declarant_ID)
                negative_voters_string = negative_voters_string + decl.declarant_ID + ", "

        if my_decl == None:
            string=(
                    ("<"+ self.ID + "> " + auction.ID).center(78,"-") + "\n" +
                    "\t I haven't declared anything" + "\n" +
                    "\t Performer Decl: %s, %s, %s" 
                        %(performer_decl.declarant_ID, performer_decl.performer_ID, performer_decl.perceived_outcome) + "\n" + 
                    "\t Agents that declared True:" + positive_voters_string.strip(", ") +"\n" +
                    "\t Agents that declared False:" + negative_voters_string.strip(", ") + "\n" +
                    ("").center(78,"-")
                   )
        else:
            string=(
                    ("<"+ self.ID + "> " + auction.ID).center(78,"-") + "\n" +
                    "\t My Decl: %s, %s, %s" 
                        %(my_decl.declarant_ID, my_decl.performer_ID, my_decl.perceived_outcome) + "\n" +
                    "\t Performer Decl: %s, %s, %s" 
                        %(performer_decl.declarant_ID, performer_decl.performer_ID, performer_decl.perceived_outcome) + "\n" + 
                    "\t Agents that declared True:" + positive_voters_string.strip(", ") +"\n" +
                    "\t Agents that declared False:" + negative_voters_string.strip(", ") + "\n" +
                    ("").center(78,"-")
                   )
        self.SAFE_log(string)
        
        if auction.verifier: 
            trust_metric = self.GetTrustMetrics(auction.ID, auction.winner)  
            if my_decl.perceived_outcome == "True":
                trust_metric.store_results(1.0)
            else :
                trust_metric.store_results(0.0)

        vtw_weight = float(len(positive_voters) - len(negative_voters))
        n_agents = float(len(self.verif_decls[auction.ID]))

        for agent_ID in positive_voters:
            trust_metric = self.GetTrustMetrics(auction.ID, agent_ID)
            vtw_n = vtw_weight/n_agents
            trust_metric.update_vtw_summ(vtw_n)
            #self.print_term(agent_ID + " Y" + str(len(trust_metric.vtw_summ)))

        for agent_ID in negative_voters:
            trust_metric = self.GetTrustMetrics(auction.ID, agent_ID)
            vtw_n = - (vtw_weight/n_agents)
            trust_metric.update_vtw_summ(vtw_n)

        for agent_ID in self.trust_metric_db.keys():
            for trust_metric in self.trust_metric_db[agent_ID].values():
                trust_metric.update_metrics(auction.ID, auction.winner, my_decl.perceived_outcome, self.mode)
        
        #Finally, update the number of total auctions in order to keep synchronized the graphs
        self.n_tot_auctions += 1
        
        if self.adapter_IP != None:
            decl_data = self.ID + "|DECLARING|" 
            for agent_ID in positive_voters:
                decl_data += agent_ID + " thinks that the task was a success"
                trust_metric = self.GetTrustMetrics(auction.ID, agent_ID)
                if trust_metric.ver_trustworthiness > 0:
                    if trust_metric.ver_trustworthiness > 0.70:
                        decl_data += " and its opinion is highly reliable; "
                    else:
                        if trust_metric.ver_trustworthiness > 0.35:
                            decl_data += " and its opinion is reliable; "
                        else:
                            decl_data += " and its opinion is not much reliable; "
                else:
                    if trust_metric.ver_trustworthiness < -0.70:
                        decl_data += ", but its opinion is totally unreliable; "
                    else:
                        decl_data += ", but its opinion is unreliable; "

            for agent_ID in negative_voters:
                decl_data += agent_ID + " thinks that the task was a failure"
                trust_metric = self.GetTrustMetrics(auction.ID, agent_ID)
                if trust_metric.ver_trustworthiness > 0:
                    if trust_metric.ver_trustworthiness > 0.70:
                        decl_data += " and its opinion is highly reliable; " 
                    else:
                        if trust_metric.ver_trustworthiness > 0.35:
                            decl_data += " and its opinion is reliable; "
                        else:
                            decl_data += " and its opinion is not much reliable; "
                else:
                    if trust_metric.ver_trustworthiness < -0.70:
                        decl_data += ", but its opinion is totally unreliable; "
                    else:
                        decl_data += ", but its opinion is unreliable; "

            self.sendToAdapter(decl_data)    

    def GetTrustMetrics(self, auction_ID, agent_ID):  #Return or create a new entry in the trust metrics list
        auction_data = auction_ID.split("/")
        action_ID = auction_data[1]


        if agent_ID in self.trust_metric_db.keys():
            if action_ID in self.trust_metric_db[agent_ID].keys():
                return self.trust_metric_db[agent_ID][action_ID]
            else:    
                self.trust_metric_db[agent_ID][action_ID] = TrustMetric(action_ID, init_results=[], init_vtw_summ=[], 
                                                                           plot_syncro = self.n_tot_auctions)
                return self.trust_metric_db[agent_ID][action_ID]
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. No trust metric for agent %s STRAAANGE" %(self.ID, agent_ID) )     

    def ComputeInitMsg(self, plans_dict):
        init_string = ""
        init_msg = Initialization()
        init_msg.agent_ID = self.ID
        for event_ID in self.plans.keys():
            init_string = init_string + event_ID + ">"
            for action_ID in self.plans[event_ID]:
                init_succ_rates, init_verification_rates = self.GetRates(action_ID, self.ID)
                init_string = init_string + action_ID + "@%0.3f" %(init_succ_rates) + "|"
            init_string = init_string.strip("|") + "#"  
        init_string = init_string.strip("#")      
        init_msg.event_actions_data = init_string
        return init_msg

    def startNextSerialAuction(self, auction_ID, new_auction_state, new_auction_data):
        self.StoreNextStateData(auction_ID, new_auction_state, new_auction_data)
        self.auct_decls[auction_ID] = []   
        self.verif_decls[auction_ID] = []
        self.action_decls[auction_ID] = []
        if self.adapter_IP != None:
            data = self.ID + "|DECLARING_IMMEDIATE|" + "Hey, boys. We need someone to say " + auction_ID.split("/")[1]
            self.sendToAdapter(data)
        auction_msg = Auction()
        auction_msg.auction_ID = auction_ID
        auction_msg.auctioneer_ID = self.ID
        auction_msg.state = "AD_START"
        self.auction_pub.publish(auction_msg) 

    def StoreNextStateData(self, ID, next_state, data =[]):
        with self.lock_us:
            if ID not in self.next_auctions_state:
                self.next_auctions_state[ID] = []
                self.next_auctions_state[ID].append([next_state, data])
            else:                    
                if next_state == "TO_DELETE":
                    event_ID = ID.split("/")[0] + "/" + ID.split("/")[2]
                    if event_ID in self.serial_auctions.keys():
                        serial_actions_list = self.serial_auctions[event_ID]
                        if len(serial_actions_list) > 0:
                            next_serial_action_data = serial_actions_list.pop(0)
                            self.startNextSerialAuction(next_serial_action_data[0], next_serial_action_data[1], next_serial_action_data[2])
                        else:
                            del self.serial_auctions[event_ID]
                    del self.next_auctions_state[ID]
                else:
                    self.next_auctions_state[ID].append([next_state, data])
                                #self.print_term(len(next_states[auct_ID]))   
                      
    def UpdateState(self):
        next_states = dict()
        next_states.update(self.next_auctions_state) 
        for auct_ID in next_states: 
            if len(next_states[auct_ID]) > 0:  
                temp = next_states[auct_ID].pop(0)

                if "Wait" in temp[0] and len(next_states[auct_ID]) > 0 :
                    ns = self.current_auctions_state[auct_ID]
                    skipped_state = temp[0]
                    temp = next_states[auct_ID].pop(0)
                    ns.state_history = ns.state_history + skipped_state.split(":")[0] + ">" 
                    self.current_auctions_state[auct_ID] = ns

                state = temp[0] 
                next_state_data = temp[1] 
                if state == "1:AD_START":
                    new_auction = Auction_Data(next_state_data[0], next_state_data[1], state)
                    if self.ID == next_state_data[1]: # if the agent is the auctioneer it will start the count down
                        ka_string = ""
                        for agent in self.known_agents: 
                            ka_string = ka_string + agent + ", " 
                        new_auction.expected_participants = ka_string.strip(", ")
                        new_auction.auct_decl_timeout = rospy.Time.now() + rospy.Duration(self.timeout)

                    new_auction.state_history = new_auction.state_history + new_auction.state.split(":")[0] + ">" 
                    #self.print_term(self.identetor("%s S[%s]->[%s]" %(self.ID, auct_ID, ns.state_history))) 
                    self.current_auctions_state[auct_ID] = new_auction 

                elif state == "2:Wait_AD_END":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns  
                        
                elif state == "3:AD_END":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.winner = next_state_data[0]
                    ns.expected_participants = next_state_data[1]
                    ns.verifying_agents = next_state_data[2]
                    ns.verifier = next_state_data[3]
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns
                    
                elif state == "4:Wait_Act_START":
                    ns = self.current_auctions_state[auct_ID]
                    if self.ID == ns.winner:
                        ns.act_decl_timeout = rospy.Time.now() + rospy.Duration(self.timeout)
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns
                    
                elif state == "5:Act_PERFORM":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns
                
                elif state == "5:Wait_Act_END":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns

                elif state == "5:Act_VERIFICATION":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns

                elif state == "6:Act_END":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns
                    
                elif state == "7:Wait_RD":
                    ns = self.current_auctions_state[auct_ID]
                    ns.res_decl_timeout = rospy.Time.now() + rospy.Duration(self.timeout)  
                    ns.state = state
                    ns.state_history = ns.state_history + state.split(":")[0] + ">"
                    self.current_auctions_state[auct_ID] = ns

                elif state == "END":
                    ns = self.current_auctions_state[auct_ID]
                    ns.state = state
                    ns.state_history = ns.state_history + state + ">"
                    #self.print_term(self.identetor("%s S[%s]-> [%s]" %(self.ID, auct_ID, ns.state_history)))
                    self.auction_database[auct_ID] = ns
                    self.StoreNextStateData(auct_ID, "TO_DELETE")
                    del self.current_auctions_state[auct_ID]
                
                else:
                    ns = self.current_auctions_state[auct_ID]
                    ns.state_history = ns.state_history + state + ">" 
                    self.print_term(self.identetor("%s ERROR. Unrecognized state %s. %s -> %s" %(self.ID, state, auct_ID, ns.state_history))) 
                    self.current_auctions_state[auct_ID] = ns 
                    
    def GetDeclarants(self, decls, auction_ID):
        declaring_agents = ""
        declaring_agents_list = []
        #Get the IDs of the agents that sent a declaration
        #self.print_term(self.identetor("GD" + str(len(decls)))
        sorted_decls = sorted(decls, key=self.getDeclarantID)

        for decl in sorted_decls:
            if decl.declarant_ID not in declaring_agents_list:
                declaring_agents = declaring_agents + decl.declarant_ID + ", "
                declaring_agents_list.append(decl.declarant_ID)
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. [%s] is already in the list of [%s]" %(self.ID, decl.declarant_ID, decl.auction_ID))
        declaring_agents = declaring_agents.strip(", ")      
        #self.print_term(self.identetor("%s Declarants for [%s] until now: %s" %(self.ID, auction_ID, declaring_agents)))
        return declaring_agents 

    def GetResDeclaration(self, auction_ID, agent_ID):
        res_decl = None
        for decl in self.verif_decls[auction_ID]:
            if decl.declarant_ID == agent_ID:
                res_decl = decl
                break
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Declaration of [%s] not found for [%s]" %(self.ID, agent_ID, auction_ID))
        return res_decl
        
    def AddNewAgentRates(self, new_agent_ID, new_agent_data):  
        with self.lock_init:
            #Adding the new agent to the known agents list
            self.updated_ka_list.append(new_agent_ID)
            self.updated_ka_list.sort() #Keep the list sorted
            #self.updated_tm[new_agent_ID] = TrustMetric(new_agent_ID)
            self.trust_metric_db[new_agent_ID] = {}

            ## ADDING ENTRY FOR NEW AGENT IN KNOWN ACTIONS

            #for each action entry in the actions dict, the agent adds an entry for the new agent to the rates list
            for action_ID in self.updated_actions_dict.keys():
                new_rates = self.updated_actions_dict[action_ID]
                temp = []
                #The chance for the agent of verifying the performances in the given action of the new agents is the same as 
                #the agent verifying its own performances in the given action
                ver_rates = self.GetRates(action_ID, self.ID)[1]
                for rate in new_rates:
                    if ver_rates != rate.verification_rates:
                        print("%s>>>>>>>>>>>>>>>ERROR: VERIFICATION RATES MUST BE THE SAME FOR THE SAME ACTION"  %self.ID)
                    temp.append(rate.performer_ID)
                #Adding to each known action_ID an entry for the new agent rates
                if new_agent_ID not in temp:
                    new_rates.append(Rates(new_agent_ID,-1.0, ver_rates)) 
                    new_rates.sort(key=self.getPerformerID)
                else:
                    print("%s>>>>>>>>>>>>>>>ERROR: Agent %s rates are already in the list" %(self.ID, new_agent_ID))    
                self.updated_actions_dict[action_ID] = new_rates
       
            ## UPDATING ACTIONS/PLAN DATABASE WITH NEW ACTIONS/PLAN DATA
            all_actions = new_agent_data.split("#")
            for string_event_action in all_actions:
                event_action_data = string_event_action.split(">")
                new_event_ID = event_action_data[0]
                new_action_data = event_action_data[1].split("|")   
                new_agent_action_rates = {}

                ## UPDATING THE ACTIONS DATABASE

                #Check if there is any new action in the plans sent by the new agent
                for action_data in new_action_data:
                    action_ID, new_agent_rates = action_data.split("@")
                    new_agent_action_rates[action_ID] = new_agent_rates

                    #Adding an entry for the actions and entries for rates of each known agent
                    if action_ID not in self.updated_actions_dict.keys():
                        new_rates = []
                        for agent_ID in self.updated_ka_list:
                            new_rates.append(Rates(agent_ID, -1.0, [-1.0 , -1.0]))
                        self.updated_actions_dict[action_ID] = new_rates 

                ##SAVING THE DECLARED SUCCESS RATES OF THE NEW AGENT
                for action_ID in new_agent_action_rates.keys():
                    self.SetInitRates(action_ID, new_agent_ID, new_agent_action_rates[action_ID])

                ## UPDATING THE PLANS DATABASE
                for event_ID in self.updated_plans_dict.keys():
                    # Check if the event_ID is already in the agent plans dict
                    if event_ID == new_event_ID:
                        #If yes, the agent checks if there are new actions in the already known event_ID
                        for action_ID in new_agent_action_rates.keys():
                            #Check if the new action_ID is not in the plans dict and append it
                            if action_ID not in self.updated_plans_dict[event_ID]:   
                                self.updated_plans_dict[event_ID].append(action_ID)
                                self.SAFE_log("> [%s] told me about its action [%s] during event [%s]" %(new_agent_ID, action_ID, event_ID))                   
                        break
                else:
                    #If no, the new event and the related actions are added to the plans dict
                    actions_string = ""
                    new_actions_list = []
                    for action_ID in new_agent_action_rates.keys():
                        new_actions_list.append(action_ID)
                        actions_string +=  action_ID + ", "
                    self.updated_plans_dict[new_event_ID] = new_actions_list
                                      
                    self.SAFE_log("> [%s] told me about its actions [%s] during the new event [%s]" 
                                    %(new_agent_ID, actions_string.strip(", "), new_event_ID)  )
                            
    def RatesToString(self, action_ID): #Method to cast a Rates() list into a string for the Auction Declaration msg
        rates_string = ""
        decl_data = ""
        best_rate_lb = []

        if self.mode == "BCIC":
            rates_string, best_rate_lb = self.BCIC_RatesToString(action_ID)                
        elif self.mode == "WINDOW":
            rates_string, best_rate_lb = self.WINDOW_RatesToString(action_ID) 
        elif self.mode == "BOOT":
            rates_string, best_rate_lb = self.BOOT_RatesToString(action_ID)
        elif self.mode == "TEST":
            rates_string, best_rate_lb = self.TEST_RatesToString(action_ID)
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. MODE NOT RECOGNIZED" %(self.ID))            
        
        decl_data = self.compute_decl_data(action_ID, best_rate_lb)
        

        return rates_string.strip("/"), decl_data

    def compute_BCI_bounds(self, tr_mtr, no_warning = True):
        n_succ = 0.0

        if len(tr_mtr.results) > 0:
            obs_rel = tr_mtr.reliability
            for result in tr_mtr.results:
                n_succ += result
                            # of succ   # of trials               #confidence 
            params = Params(k=n_succ, n = len(tr_mtr.results), confi_perc = BCIC_CONFIDENCE)
            lower_p, upper_p = exact(params)
            if verify_interval_of_p(params ,lower_p, upper_p, -14, verbose=1) != 0:
                print("%s>>>>>>>>>>>>>>>ERROR. BCIC with these parameters [%d|%d|%d] may be unreliable" 
                                                        %(self.ID, params.k, params.n, params.confi_perc))
        else:
            if no_warning:
                print("%s>>>>>>>>>>>>>>>WARNING. Computing BCIC with n = 0" %(self.ID))

            if self.disposition == "Optimist":
                obs_rel = 1.0
            elif self.disposition == "Pessimist":
                obs_rel = 0.0
            elif self.disposition == "Realist":
                obs_rel = 0.0
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

            lower_p = 0.0
            upper_p = 1.0
        
        return obs_rel, lower_p, upper_p

    def BCIC_RatesToString(self, action_ID):
        rates_string = ""
        debug_rates_string = ""
        best_rate = 0.0
        best_rate_lb = []
        obs_rel = 0.0
        lower_p = 0.0
        upper_p = 0.0

        for rate in self.actions[action_ID]:
            if action_ID in self.trust_metric_db[rate.performer_ID].keys():
                tr_mtr = self.trust_metric_db[rate.performer_ID][action_ID]
                #If the agents has observed at least once the other agent performing the action, it computes BCI
                if len(tr_mtr.results) > 0:
                    obs_rel, lower_p, upper_p = self.compute_BCI_bounds(tr_mtr)
                    if self.disposition == "Optimist":
                        #storing the best reliability in order to notify the user with a proper declaration
                        if upper_p > best_rate:
                            best_rate_lb = []
                            best_rate = upper_p
                        if upper_p == best_rate:
                            best_rate_lb.append(rate.performer_ID)
                    elif self.disposition == "Pessimist":
                        #storing the best reliability in order to notify the user with a proper declaration
                        if lower_p > best_rate:
                            best_rate_lb = []
                            best_rate = lower_p
                        if lower_p == best_rate:
                            best_rate_lb.append(rate.performer_ID)
                    elif self.disposition == "Realist":
                        #storing the best reliability in order to notify the user with a proper declaration
                        if obs_rel > best_rate:
                            best_rate_lb = []
                            best_rate = obs_rel
                        if obs_rel == best_rate:
                            best_rate_lb.append(rate.performer_ID)
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Something went wrong with my dispostion" %(self.ID))

                    rates_string += ("%s;%0.3f|%0.3f|%0.3f;%0.3f;%0.3f/" 
                                    %(rate.performer_ID, obs_rel, lower_p, upper_p , rate.verification_rates[0], rate.verification_rates[1]))
                    debug_rates_string += ("%s;OBS:%0.3f|LOW:%0.3f|UP:%0.3f;%0.3f;%0.3f\n" 
                                        %(rate.performer_ID, obs_rel, lower_p, upper_p , rate.verification_rates[0], rate.verification_rates[1])
                                        + " "*len(self.ID + " DECLARES: "))
                else:  
                    if rate.performer_ID == self.ID:
                        #storing the best reliability in order to notify the user with a proper declaration
                        if rate.success_rate > best_rate:
                            best_rate_lb = []
                            best_rate = rate.success_rate
                        if rate.success_rate == best_rate:
                            best_rate_lb.append(rate.performer_ID)

                        rates_string += ("%s;%0.3f|%0.3f|%0.3f;%0.3f;%0.3f/" 
                                        %(rate.performer_ID, rate.success_rate, 0.0, 1.0, rate.verification_rates[0], rate.verification_rates[1]))
                        debug_rates_string += ("%s;OBS:%0.3f|LOW:%0.3f|UP:%0.3f;%0.3f;%0.3f\n" 
                                                %(rate.performer_ID, rate.success_rate, 0.0, 1.0, rate.verification_rates[0], rate.verification_rates[1])
                                                + " "*len(self.ID + " DECLARES: "))   
                    
            elif rate.performer_ID == self.ID:

                #storing the best reliability in order to notify the user with a proper declaration
                if rate.success_rate > best_rate:
                    best_rate_lb = []
                    best_rate = rate.success_rate
                if rate.success_rate == best_rate:
                    best_rate_lb.append(rate.performer_ID)

                rates_string += ("%s;%0.3f|%0.3f|%0.3f;%0.3f;%0.3f/" 
                                %(rate.performer_ID, rate.success_rate, 0.0, 1.0, rate.verification_rates[0], rate.verification_rates[1]))
                debug_rates_string += ("%s;OBS:%0.3f|LOW:%0.3f|UP:%0.3f;%0.3f;%0.3f\n" 
                                       %(rate.performer_ID, rate.success_rate, 0.0, 1.0, rate.verification_rates[0], rate.verification_rates[1])
                                       + " "*len(self.ID + " DECLARES: "))    
            
        print(self.ID + " DECLARES: " + debug_rates_string +"\n")
        self.SAFE_log(self.ID + " DECLARES: " + debug_rates_string)
        return rates_string, best_rate_lb 

    def WINDOW_RatesToString(self, action_ID):
        rates_string = ""
        best_rate = 0.0
        best_rate_lb = []
        debug_string = self.ID + " DECLARES:\n"
        for rate in self.actions[action_ID]:
            if action_ID in self.trust_metric_db[rate.performer_ID].keys():
                tr_mtr = self.trust_metric_db[rate.performer_ID][action_ID]
                if len(tr_mtr.results) < WINDOW_LENGTH:
                    debug_string +=("\t not enough data about %s -> sending its first declared rate: %0.3f\n" 
                            %(rate.performer_ID, rate.success_rate))

                    #storing the best reliability in order to notify the user with a proper declaration
                    if rate.success_rate > best_rate:
                        best_rate_lb = []
                        best_rate = rate.success_rate
                    if rate.success_rate == best_rate:
                        best_rate_lb.append(rate.performer_ID)
                        
                    rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                    %(rate.performer_ID, rate.success_rate, rate.verification_rates[0], rate.verification_rates[1]))
                
                else:
                    debug_string +=("\t sending %s reliability in %s: %0.3f\n" %(rate.performer_ID, action_ID, tr_mtr.reliability))

                    #storing the best reliability in order to notify the user with a proper declaration
                    if tr_mtr.reliability > best_rate:
                        best_rate_lb = []
                        best_rate = tr_mtr.reliability
                    if tr_mtr.reliability == best_rate:
                        best_rate_lb.append(rate.performer_ID)

                    rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                    %(rate.performer_ID, tr_mtr.reliability, rate.verification_rates[0], rate.verification_rates[1]))
            else:
                debug_string +=(" first time seeing %s partecipating to an auction for this action -> sending its first declared rate %0.3f\n" 
                        %(rate.performer_ID, rate.success_rate))

                #storing the best reliability in order to notify the user with a proper declaration
                if rate.success_rate > best_rate:
                    best_rate_lb = []
                    best_rate = rate.success_rate
                if rate.success_rate == best_rate:
                    best_rate_lb.append(rate.performer_ID)

                rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                %(rate.performer_ID, rate.success_rate, rate.verification_rates[0], rate.verification_rates[1]))
        print(debug_string)
        return rates_string, best_rate_lb 
    
    def BOOT_RatesToString(self, action_ID):
        rates_string = ""
        best_rate = 0.0
        best_rate_lb = []
        debug_string = self.ID + " DECLARES:\n"
        for rate in self.actions[action_ID]:
            if action_ID in self.trust_metric_db[rate.performer_ID].keys():
                tr_mtr = self.trust_metric_db[rate.performer_ID][action_ID]
                if len(tr_mtr.results) < BOOT_WINDOW_LENGTH:
                    debug_string +=("\t not enough data about %s -> sending its first declared rate: %0.3f\n" 
                            %(rate.performer_ID, rate.success_rate))

                    #storing the best reliability in order to notify the user with a proper declaration
                    if rate.success_rate > best_rate:
                        best_rate_lb = []
                        best_rate = rate.success_rate
                    if rate.success_rate == best_rate:
                        best_rate_lb.append(rate.performer_ID)
                        
                    rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                    %(rate.performer_ID, rate.success_rate, rate.verification_rates[0], rate.verification_rates[1]))
                
                else:
                    debug_string +=("\t sending %s reliability in %s: %0.3f\n" %(rate.performer_ID, action_ID, tr_mtr.reliability))

                    #storing the best reliability in order to notify the user with a proper declaration
                    if tr_mtr.reliability > best_rate:
                        best_rate_lb = []
                        best_rate = tr_mtr.reliability
                    if tr_mtr.reliability == best_rate:
                        best_rate_lb.append(rate.performer_ID)

                    rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                    %(rate.performer_ID, tr_mtr.reliability, rate.verification_rates[0], rate.verification_rates[1]))
            else:
                debug_string +=(" first time seeing %s partecipating to an auction for this action -> sending its first declared rate %0.3f\n" 
                        %(rate.performer_ID, rate.success_rate))

                #storing the best reliability in order to notify the user with a proper declaration
                if rate.success_rate > best_rate:
                    best_rate_lb = []
                    best_rate = rate.success_rate
                if rate.success_rate == best_rate:
                    best_rate_lb.append(rate.performer_ID)

                rates_string += ("%s;%0.3f;%0.3f;%0.3f/" 
                                %(rate.performer_ID, rate.success_rate, rate.verification_rates[0], rate.verification_rates[1]))
        print(debug_string)
        return rates_string, best_rate_lb 

    def TEST_RatesToString(self, action_ID):
        rates_string = ""
        best_rate = 0.0
        best_rate_lb = []

        for rate in self.actions[action_ID]:
            if action_ID in self.trust_metric_db[rate.performer_ID].keys() and self.TEST_phase_on == False:
                tr_mtr = self.trust_metric_db[rate.performer_ID][action_ID]
                
                #storing the best reliability in order to notify the user with a proper declaration
                if tr_mtr.reliability > best_rate:
                    best_rate_lb = []
                    best_rate = tr_mtr.reliability
                if tr_mtr.reliability == best_rate:
                    best_rate_lb.append(rate.performer_ID)
        
                rates_string += ("%s;%0.3f;%0.3f;%0.3f/" %(rate.performer_ID, tr_mtr.reliability , rate.verification_rates[0], rate.verification_rates[1]))
            else:
                #storing the best reliability in order to notify the user with a proper declaration
                if rate.success_rate > best_rate:
                    best_rate_lb = []
                    best_rate = rate.success_rate
                if rate.success_rate == best_rate:
                    best_rate_lb.append(rate.performer_ID)

                rates_string += ("%s;%0.3f;%0.3f;%0.3f/" %(rate.performer_ID,rate.success_rate, rate.verification_rates[0], rate.verification_rates[1]))
        

        return rates_string, best_rate_lb 

    def compute_decl_data(self, action_ID, best_rate_lb):
        decl_data = ""

        if len(best_rate_lb) == 1:
            best_performer = best_rate_lb[0]
            if best_performer != self.ID:
                decl_data = ("%s may be better than me at saying %s." %(best_performer,action_ID))
            else:
                decl_data = ("No one is better than me at saying %s" %(action_ID))

        else:
            if self.ID not in best_rate_lb:
                best_peformers_string =""
                for agent in best_rate_lb:
                    best_peformers_string += agent + ", "
                decl_data = ("%s may be better than me at saying %s." 
                                %(best_peformers_string.strip(", "),action_ID))              
            else:
                best_peformers_string =""
                for agent in best_rate_lb:
                    if agent != self.ID:
                        best_peformers_string += agent + ", "
                decl_data = ("My friends %s may be as good as me at saying %s" 
                                %(best_peformers_string, action_ID))

        return decl_data

    def StringToRates(self,rates_string): #Method to cast the string received in the Auction Declaration msg into a Rates() list
        declared_rates = []
        agent_rates = rates_string.split("/") 
        for agent_rate in agent_rates:
            temp = agent_rate.split(";")
            performer_ID = temp[0]
            declared_ver_rates = [float(temp[2]), float(temp[3])]
            if self.mode == "BCIC":
                declared_succ_rate, lower_succ_rate, upper_succ_rate = temp[1].split("|")
                decl_rate = Rates(performer_ID, float(declared_succ_rate), declared_ver_rates, float(lower_succ_rate), float(upper_succ_rate))
            else:
                declared_succ_rate = float(temp[1])
                decl_rate = Rates(performer_ID, declared_succ_rate, declared_ver_rates)

            declared_rates.append(decl_rate)
            
        return declared_rates[:]

    def GetRates(self, action_ID, agent_ID):
        agent_rates = None
        if action_ID in self.actions.keys():
            all_agent_rates = self.actions[action_ID]
            for rates in all_agent_rates:
                if rates.performer_ID == agent_ID:
                    agent_rates = rates
                    break
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. No rates found for agent [%s]" %(self.ID, agent_ID)) 
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. No entry found for action [%s]" %(self.ID, action_ID)) 

        if agent_rates == None:   
            print("%s>>>>>>>>>>>>>>>ERROR, NO RATES FOR ACTION %s PERFORMED BY THE GIVEN AGENT %s" %(self.ID, action_ID, agent_ID))

        return agent_rates.success_rate, agent_rates.verification_rates

    def SetInitRates(self, action_ID, agent_ID, new_succ_rate):
        if action_ID in self.actions.keys():
            all_agent_rates = self.actions[action_ID]
            for rates in all_agent_rates:
                if rates.performer_ID == agent_ID:
                    rates.success_rate = float(new_succ_rate)
                    break
            else:
                print("%s>>>>>>>>>>>>>>>ERROR. No rates found for agent [%s]" %(self.ID, agent_ID)) 
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. No entry found for action [%s]" %(self.ID, action_ID))

    def print_trust_db_size(self):  
        string = (self.ID + " " + str(len(self.trust_metric_db)))
        for ag_ID in sorted(self.trust_metric_db):
            string += " " + (str(len(self.trust_metric_db[ag_ID])))
        self.print_term(string)

    def print_everything_about_trust(self):
        string = (self.ID + " Dim(" + str(len(self.trust_metric_db)) + ")\n")
        for ag_ID in sorted(self.trust_metric_db):
            string += "Trust in -->" + ag_ID + "Dim(" + (str(len(self.trust_metric_db[ag_ID])) + "): ")
            for act_ID in sorted(self.trust_metric_db[ag_ID]):
                obj = self.trust_metric_db[ag_ID][act_ID]
                string += "%s(%d -> %0.3f / %d -> %0.3f) " %(act_ID, len(obj.results), obj.reliability, len(obj.vtw_summ), obj.ver_trustworthiness)
            string += "\n"
        print(string)

    def log_auction_declarations(self, auction): #Print method for Auction Declaration (table)
        data_string = []
        auct_decls_string = ""

        auct_decls_string += ("Auction declarations received for %s" %(auction.ID)).center(129, "_") + "\n"

        for decl in self.auct_decls[auction.ID]:
            for rate in decl.rates_list:
                eVerRateTT = rate.verification_rates[0]
                eVerRateFT = 1.0 - rate.verification_rates[0]
                eVerRateFF = rate.verification_rates[1]
                eVerRateTF = 1.0 - rate.verification_rates[1]

                data = [decl.declarant_ID, rate.performer_ID , rate.success_rate, eVerRateTT, eVerRateFT, eVerRateFF, eVerRateTF, decl.auction_ID]
                data_string.append(data)
        
        auct_decls_string += ( R"| Declarant | Performer | eSucc% | eVer%(T|T) | eVer%(F|T) | eVer%(F|F) | eVer%(T|F) | Auction ID                               |" ) + "\n"
        data_string.sort(key=self.getElemZero)
        for x in data_string:
            auct_decls_string += ("|%s|%s|   %0.3f|       %0.3f|       %0.3f|       %0.3f|       %0.3f|%s|\n" 
                           %(self.NameCutter(x[0],11), self.NameCutter(x[1],11), x[2], x[3], x[4], x[5], x[6], self.NameCutter(x[7],42))) 
        
        auct_decls_string +=(  "'–––––––––––'–––––––––––'––––––––'––––––––––––'––––––––––––'––––––––––––'––––––––––––'––––––––––––––––––––––––––––––––––––––––––'\n" )

        return auct_decls_string +"\n"

    def log_known_agents(self): #Print method for the known agent list 
        string = "KNOWN AGENTS: \n"
        for x in self.known_agents:
            string += x + ", "
        return string.strip(", ") + "\n"

    def log_actions(self): #Log method for Action list (table)
        log_actions_string = ("\n " + self.ID + " ACTIONS LIST START ").center(78,"_")+ "\n"
        for event_ID in self.plans.keys():
            log_actions_string +=  (" " + event_ID + " - " + self.ID + " ").center(78,"=") +"\n"
            for action_ID in self.plans[event_ID]:
                action_string = '{:-<78}'.format(action_ID)
                log_actions_string += action_string +"\n"
                data_string = []
                
                for rate in sorted(self.actions[action_ID], key=self.getPerformerID):
                    agent_ID = rate.performer_ID 
                    eVerRateTT = rate.verification_rates[0]
                    eVerRateFT = 1.0 - rate.verification_rates[0]
                    eVerRateFF = rate.verification_rates[1]
                    eVerRateTF = 1.0 - rate.verification_rates[1]
                    data = [agent_ID, rate.success_rate, eVerRateTT, eVerRateFT, eVerRateFF, eVerRateTF ]
                    data_string.append(data)

                log_actions_string +=  "| Agents | eSucc% | eVer%(T|T) | eVer%(F|T) | eVer%(F|F) | eVer%(T|F) | " + "\n"
                for x in data_string:
                    log_actions_string +=("|%s|   %0.3f|       %0.3f|       %0.3f|       %0.3f|       %0.3f|\n" 
                                   %(self.NameCutter(x[0], 8),      x[1],         x[2],         x[3],         x[4],         x[5]) )  

                log_actions_string += '{:->78}'.format(action_ID) +"\n"
        log_actions_string += (" " + self.ID + " ACTIONS LIST END ").center(78,"_") + "\n"

        return log_actions_string    

    def log_event_actions(self, event_ID):
        actions_string = ""

        if event_ID in self.plans.keys():
            for action_ID in self.plans[event_ID]:
                actions_string +=  "[" + action_ID + "], "
        else:    
            print("%s>>>>>>>>>>>>>>>ERROR. No entry found for event [%s]" %(self.ID, event_ID))    

        self.SAFE_log("> Possible actions during event [" + event_ID + "]: " +  actions_string.strip(", "))
        
    def log_result_declarations(self, auction):
        res_decls_str =  ("RESULT VERIFICATION FOR %s" %(auction.ID) ).center(87, "_") + "\n"

        res_decls_str += ( "| Auction ID                               | Declarant | Performer | Perceived Result |\n")     

        for decl in self.verif_decls[auction.ID]:
            res_decls_str += ("|%s|%s|%s|%s|\n" %(self.NameCutter(decl.auction_ID,42), self.NameCutter(decl.declarant_ID,11), self.NameCutter(decl.performer_ID,11), 
                                             self.NameCutter(decl.perceived_outcome,18)))

        res_decls_str += ( ("").center(87, "_") + "\n")

        return res_decls_str + "\n"

    def log_trust_metrics(self): 
        string = (self.ID + "  TRUST METRICS (" + str(len(self.trust_metric_db)) + ")" ).center(78,"_") + "\n"
        for ag_ID in sorted(self.trust_metric_db):
            #string += "Trust in -->" + ag_ID + "(" + (str(len(self.trust_metric_db[ag_ID])) + "): ")
            string += "$" + ag_ID[0] + "_" + ag_ID[1] +"$ & "
            for act_ID in sorted(self.trust_metric_db[ag_ID]):
                obj = self.trust_metric_db[ag_ID][act_ID]
                n_succ = 0
                n_fail = 0
                for result in obj.results:
                    if result == 1.0:
                        n_succ += 1
                    elif result == 0.0:
                        n_fail += 1
                    else:
                        print("%s>>>>>>>>>>>>>>>ERROR. Unrecognized result" %(self.ID)) 
                string += ("%0.2f (%d; %d) & %0.2f (%d) & " 
                          %(obj.reliability, n_succ, n_fail ,obj.ver_trustworthiness, len(obj.vtw_summ)))
            string = string.strip("& ") + R"\\" + "\n" 

        string += (self.ID + "  TRUST METRICS").center(78,"_") + "\n"
        return string
    
    def print_trust_metrics(self): 
        
        string = (self.ID + "  TRUST METRICS").center(78,"_") + "\n"
        for agent_ID in sorted(self.trust_metric_db.keys()):
            string += ("<TRUST IN " + agent_ID)
            for action_ID, trust_metric in self.trust_metric_db[agent_ID].items():
                string += "\t" + action_ID + "     "
                string += ("> "+ "REL:%0.3f" %(trust_metric.reliability)
                               + "VTW:%0.3f" %(trust_metric.ver_trustworthiness)
                               + "\n") 
            string += "\n"
        string += (self.ID + "  TRUST METRICS").center(78,"_")
        self.print_term(string)

    def save_log(self):

        if self.experiment_data_file_path == "":
            now = datetime.now()
            self.experiment_data_file_path = Path(__file__).parent / ('Experiments/Exp_' + now.strftime("%d_%m_%Y_%H_%M"))
        filename = self.experiment_data_file_path / self.ID 

        log_str = self.SAFE_log(clean=True)
        cb_log_str = self.SAFE_callback_logger(clean=True)
        now = datetime.now()
        log_str += ( "\n\n" +  ("AT " + now.strftime("%d_%m_%Y_%H_%M") + " ").center(78,"#") + "\n")
        log_str += ( "AGENT ID: <" + self.ID + ">" + "\n")
        log_str += self.log_known_agents()
        log_str += self.log_actions()
        log_str += self.log_trust_metrics()

        log_str += (("").center(78,"#") + "\n")
        log_str += ("\n")
        log_str += ( "ONGOING AUCTIONS :" + "\n")
        if len(self.current_auctions_state) == 0:
            log_str += ( "\t NONE "+ "\n")
        else:
            for auction in self.current_auctions_state.values():
                log_str += ( auction.ID + "\t STATE:" + auction.state + "\n")
                log_str += ( " ONGOING AUCTION DECLARATIONS : ")
                if auction.ID in self.auct_decls:
                    log_str += ( "\n")
                    log_str + self.log_auction_declarations(auction)
                else: 
                    log_str += ( "NONE > OK!" + "\n")

                log_str += ( " ONGOING ACTION DECLARATIONS : ")

                if auction.ID in self.action_decls:
                    log_str += ( "\n")
                    for x in self.action_decls[auction.ID]:
                        log_str += ( x.declarant_ID + " " + x.state + "\n")
                else: 
                    log_str += ( "NONE > OK!" + "\n")

                log_str += ( " ONGOING VERIFICATION DECLARATIONS : ")
                if auction.ID in self.verif_decls:   
                    log_str += ("\n")
                    log_str += self.log_result_declarations(auction)
                else: 
                    log_str += ( "NONE > OK!" + "\n")

                log_str += ("\n")
        

        log_str += ( "\nCOMPLETED AUCTIONS: " + "\n")

        if len(self.auction_database) == 0:
            log_str += ( "\t NONE ?!?!?!?" + "\n")
        else: 
            for auction in self.auction_database.values():
                if auction.state_history != "1>2>3>4>5>6>7>END>":
                    print("ERROR IN STATE HISTORY!!!!!!!")
                    log_str += ( auction.ID + "\t STATE:" + auction.state_history + "\n")
                    
                    log_str += ( " REMAINING AUCTION DECLARATIONS : " + "\n")
                    if auction.ID in self.auct_decls:
                        log_str += ( "\n")
                        log_str += self.log_auction_declarations(auction)
                    else: 
                        log_str += ( "NONE > OK!"+ "\n")

                    log_str += ( " REMAINING ACTION DECLARATIONS : ")

                    if auction.ID in self.action_decls:
                        log_str += ( "\n")
                        for x in self.action_decls[auction.ID]:
                            log_str += ( x.declarant_ID + " " + x.state + "\n")
                    else: 
                        log_str += ( "NONE > OK!" + "\n")

                    log_str += ( " REMAINING VERIFICATION DECLARATIONS : "+ "\n")
                    if auction.ID in self.verif_decls:   
                        log_str += ("\n")
                        log_str += self.log_result_declarations(auction)
                    else: 
                        log_str += ( "NONE > OK!" + "\n")

                    log_str += ("\n")
                    break
            else:
                log_str += "%d Auctions completed without errors" %len(self.auction_database)


        log_str += ( "\nCALLBACK LOG : " + "\n")
        log_str += ( cb_log_str + "\n")
        
        while True:
            try:
                if not os.path.exists(self.experiment_data_file_path):
                    os.makedirs(self.experiment_data_file_path)
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise   
                # time.sleep might help here
                self.print_term(
                      "##################################################################################################################\n"+
                      "#######################################LOG: IT HAPPENED, BUT PREVENTED ###########################################\n"+
                      "##################################################################################################################\n")
                pass
        
        with open(filename, "a") as f:
            f.write(log_str)

    def save_plot_data(self): 
        plot_file_string = ""

        if self.experiment_data_file_path == "":
            now = datetime.now()
            self.experiment_data_file_path = Path(__file__).parent / ('/Experiments/Exp_' + now.strftime("%d_%m_%Y_%H_%M"))
        filename = self.experiment_data_file_path / ('plot_data/' + self.ID) 

        plot_file_string = "#" + self.ID + "\n"
        row_string = ""
        col_string = ""
        for agent_ID in sorted(self.trust_metric_db.keys()):
            temp = agent_ID
            for tr_mtr in self.trust_metric_db[agent_ID].values():
                temp += "|" + tr_mtr.ID 
            col_string += self.NameCutter(temp, 20*len(self.trust_metric_db[agent_ID])+2)
        for i in range(0, self.n_tot_auctions):
            row_string += "!" + str(i) + "|"
            for agent_ID in sorted(self.trust_metric_db.keys()):
                for tr_mtr in self.trust_metric_db[agent_ID].values():
                    row_string += self.NameCutter("%s; %0.3f; %0.3f"%(tr_mtr.rel_plt_data[i][1], tr_mtr.rel_plt_data[i][0], tr_mtr.vtw_plt_data[i]), 60) + "|"
            row_string += "\n"
        plot_file_string += ("$" + col_string + "\n" + row_string)

        while True:
            try:
                if not os.path.exists(self.experiment_data_file_path / 'plot_data'):
                    os.makedirs(self.experiment_data_file_path / 'plot_data')
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise   
                # time.sleep might help here
                self.print_term(
                      "##################################################################################################################\n"+
                      "########################################PD: IT HAPPENED, BUT PREVENTED ###########################################\n"+
                      "##################################################################################################################\n")
                pass
        
        with open(filename, "w") as f:
            f.write(plot_file_string)
            
    def NameCutter(self, string, cut_length):
        new_string = ""
        if len(string) > cut_length:
            new_string = string[:cut_length]
        else:
            new_string = string + " "*(cut_length-len(string))

        return new_string

    def sendToAdapter(self,data):
        if data.split("|")[1] not in ["PERFORMING" , "VERIFYING"]:
            return
        # Create a client socket
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        clientSocket.connect((self.adapter_IP,self.adapter_port_num))
        clientSocket.send(data.encode())
        dataFromClient = clientSocket.recv(1024)
        # if str(dataFromClient.decode()) == "True":
        #     data = True
        # elif str(dataFromClient.decode()) == "False":
        #     data = False
        # elif str(dataFromClient.decode()) == "ACK":
        #     pass
        # else:
        #     print("%s>>>>>>>>>>>>>>>ERROR. Data from the robot not recognized" %(self.ID))
        if str(dataFromClient.decode()) == "True":
            data = "True"
        elif str(dataFromClient.decode()) == "False":
            data = "False"
        elif str(dataFromClient.decode()) == "ACK":
            pass
        else:
            print("%s>>>>>>>>>>>>>>>ERROR. Data from the robot not recognized" %(self.ID))

        clientSocket.shutdown(socket.SHUT_RDWR)
        clientSocket.close()
        return data 

    def getPerformerID(self, elem): #Support key for sorting lists
        return elem.performer_ID

    def getAgentID(self, elem): #Support key for sorting lists
        return elem.agent_ID

    def getDeclarantID(self, elem): #Support key for sorting lists
        return elem.declarant_ID
    
    def getID(self, elem): #Support key for sorting lists
        return elem.ID

    def getElemZero(self, elem): #Support key for sorting lists
        return elem[0]

    def identetor(self, string):
        if self.identetor_flag:
            formatted_string = "|"

            while len(string) > 20:
                for x in range(0,10):
                    if x == (int(self.ID.strip("G"))-1):
                        formatted_string += string[:20] + "|"
                    else:
                        formatted_string +=" "*20 + "|"
                formatted_string += "\n|"
                string = string[20:]
            if string != "":
                string += " "*(20 - len(string))
                for x in range(0,10):
                    if x == (int(self.ID.strip("G"))-1):
                        formatted_string += string + "|"  
                    else:
                        formatted_string +=" "*20 + "|" 

            return formatted_string
        else:
            return string
        
    def SAFE_callback_logger(self, string = "", clean = False):
        if self.callback_log_on:
            with self.lock_log:
                if clean == True:
                    cb_string = self.callback_log
                    self.callback_log = ""
                    return cb_string
                else:
                    self.callback_log += (string) + "\n"
        else:
            return

    def SAFE_log(self, string = "", no_new_line = False, clean = False):
        if self.log_on:
            with self.lock_log:
                if clean == True:
                    log_str = self.log_string
                    self.log_string = ""
                    return log_str
                elif no_new_line:
                    self.log_string += string
                else:
                    self.log_string += string + "\n"
        else:
            return

    def print_term(self, string):
        if self.print_to_term_flag:
            print(string)
        else:
            pass

    def len_printer(self):
        print(self.ID + " %d-%d  %d-%d  %d-%d  \n %d  %d  %d  %d \n%d %d %d %d" 
              %(len(self.known_agents), len(self.updated_ka_list), 
                len(self.plans), len(self.updated_plans_dict), 
                len(self.actions), len(self.updated_actions_dict), 
                len(self.current_auctions_state), len(self.next_auctions_state) , len(self.serial_auctions), len(self.auction_database),
                len(self.auctions_queue), len(self.auct_decls), len(self.verif_decls), len(self.action_decls))    )

    def plot_graphs(self, experiment_path):

        RLpath = experiment_path / "plot_data"
        fig_list = []
        tr_db = dict()
        vtw_db = dict()
        x_ticks = dict()
        for filename in os.listdir(RLpath):
            with open(os.path.join(RLpath, filename), 'r') as f: # open in readonly mode
                agentX = ""
                for line in f:
                    if line[0] == "#":
                        agentX = line.strip("#")
                        agentX = agentX.strip("\n")
                        fig_list.append(agentX)
                        tr_db[agentX] = {}
                        vtw_db[agentX] = {}
                        x_ticks[agentX] = {}
                    
                    elif line[0] == "$":
                        temp_line = line.strip("$")
                        temp_line = temp_line.strip("\n")
                        agents_list = temp_line.split()
                        for tr_ag_ID in agents_list:
                            agent_ID = tr_ag_ID.split("|")[0]
                            actions_list = tr_ag_ID.split("|")[1:]
                            tr_db[agentX][agent_ID] = {}
                            vtw_db[agentX][agent_ID] = {}
                            for action_ID in actions_list:
                                tr_db[agentX][agent_ID][action_ID] = []
                                vtw_db[agentX][agent_ID][action_ID + " VTW"] = []
                    elif line[0] == "!" :
                        if line[1] == "\n":
                            continue
                        else:
                            temp = line[1:]
                            temp = temp.strip("\n")
                            temp = temp.strip("|")
                            row = temp.split("|")
                            if len(row) == 0:
                                print(row)
                            else:
                                x = row.pop(0)
                            for agent_ID in tr_db[agentX].keys():
                                for action_ID in tr_db[agentX][agent_ID].keys():
                                    if len(row) == 0:
                                        print(row)
                                    else:
                                        data = row.pop(0)    
                                    values = data.strip()
                                    if len(values.split("; ")) != 3:
                                        print(values.split("; "))
                                    else:
                                        auct_data, y1, y2  = values.split("; ")
                                    if "NONE" in auct_data:
                                        pass
                                    else:
                                        ev_ID, act_ID, auct_count, winner, perc_outcome = auct_data.split("/")
                                        if perc_outcome == "True":
                                            perc_outcome = "T"
                                        else:
                                            perc_outcome = "F"
                                        #tick = winner + "/" + perc_outcome
                                        tick = auct_count + "_" + self.NameCutter(act_ID, 5) + "/" + winner + "/" + perc_outcome
                                    x_ticks[agentX][x] = tick
                                    tr_db[agentX][agent_ID][action_ID].append([int(x), float(y1)])
                                    vtw_db[agentX][agent_ID][action_ID + " VTW"].append([int(x), float(y2)])
                        
                    else:
                        print(line)

        def plot_the_plots(ID, tr_database, vtw_db, ticks, folder_path):
            TITLESIZE = 25
            XLABELSIZE = 15 
            YLABELSIZE = 15
            LEGENDSIZE = 15

            # Setup figure and subplots
            colors = dict()
            colors.update(mcolors.CSS4_COLORS)
            chosen_colors = []
            names = ["crimson", "mediumspringgreen", "blue", "darkorange", "limegreen",  "pink", "gold", "aqua", "blue", "mediumorchid", "darkgrey" ,"black"]
                    #"navajowhite", "deeppink", "khaki", "darkgreen", "rebeccapurple", "lightcoral", "teal", "deepskyblue", "lawngreen"]
            for name in names:
                chosen_colors.append(colors[name])
            exp_db = tr_database[ID]    
            x_ticks = ticks
            f1 = figure(int(ID.strip("G")), figsize = (12, 8))#, dpi = 100)
            #f1.suptitle("%s Trust Metrics" %ID, fontsize=TITLESIZE)

            #dict of plots axis
            plt_axs = {}
            i = 0
            N_agents = len(tr_database.keys())
            nCols = int(ceil(N_agents/5)) 
            #Setting the subplot grid
            for agent_ID in sorted(tr_database.keys()):
                if N_agents <= 5:
                    plt_axs[agent_ID] = subplot2grid((N_agents,nCols), (i, 0))
                elif N_agents <= 10:
                    if i < 5:
                        plt_axs[agent_ID] = subplot2grid((5,nCols), (i, 0))
                    else:
                        plt_axs[agent_ID] = subplot2grid((5,nCols), (i-5, 1))
                else:
                    plt_axs[agent_ID] = subplot2grid((5,nCols), (i%5, int(i/5)))
                i +=1
            i = 0
            # set y-limits
            for ax in plt_axs.values():
                ax.set_ylim(-1.05,1.05)

            # Turn on grids
            for ax in plt_axs.values():
                ax.grid(True)
            
            #Extract data
            for trustee in exp_db.keys():
                i = 0
                for exp_ID, exp_plt_data in exp_db[trustee].items():
                    x = []
                    y = []
                    for elem in exp_plt_data:
                        x.append(elem[0])
                        y.append(elem[1])
                    plt_axs[trustee].plot(x, y, "-", color=chosen_colors[i], label=exp_ID + " Rel")
                    i += 1
            
            for trustee in vtw_db.keys():
                i = 0
                for vtw_ID, vtw_plt_data in vtw_db[trustee].items():
                    x = []
                    y = []
                    for elem in vtw_plt_data:
                        x.append(elem[0])
                        y.append(elem[1])

                    plt_axs[trustee].plot(x, y, "--", color=chosen_colors[i], label=vtw_ID)
                    i += 1
            i = 0

            def format_func(value, tick_number):
                N = int(np.floor(value))
                if N < 0:
                    return "NO_DATA"
                elif N > len(x_ticks)-1:
                    return ""
                else:
                    return x_ticks[str(N)]

            # set label names
            for agent_ID,ax in plt_axs.items():
                ax.set_ylabel("%s" %(agent_ID), fontsize=YLABELSIZE)
                ax.set_yticks(np.arange(-1.0, 1.2, 0.50)) # setting the ticks
                
        
            ax.set_xlabel("Auctions", fontsize=XLABELSIZE)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            f1.legend(by_label.values(), by_label.keys(), loc='upper center', fontsize = LEGENDSIZE, ncol = len(by_label.keys()) ) #bbox_to_anchor=(0.9, 1), ncol=int(ceil(len(by_label.keys())/4))
            
            while True:
                try:
                    if not os.path.exists(os.path.join(folder_path)):
                        os.makedirs(os.path.join(folder_path))
                    break
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise   
                    pass
            
            filename = "%s_Exp_Plots" %ID
            f1.savefig(os.path.join(folder_path, filename))


        folder_path = experiment_path / "graphs"
        for ID in fig_list:
            plot_the_plots(ID, tr_db, vtw_db[ID], x_ticks[ID], folder_path)


    
