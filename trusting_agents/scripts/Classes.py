#!/usr/bin/env python3

class Rate(): #Class to store the estimated succ. and verif. rates of an agent 
    def __init__(self,agent_ID,success_rate, verification_rates):
        self.agent_ID = agent_ID 
        self.success_rate = success_rate
        self.verification_rates = verification_rates

class Action(): #Class to store an action data by specifying its ID, its triggering event and the list of rates for such action for each known agent 
                #(including the agent itself) 
    
    def __init__(self,action_ID, given_rates = []):
        #action id
        self.action_ID = action_ID
        #agent-indexed list about the rates of success and verification of the action 
        self.rates = list(given_rates)


    def get_agent_rates(self,agent_ID):
        for x in self.rates:
            if x.agent_ID == agent_ID:
                verification_rates = x.verification_rates
                success_rate = x.success_rate
        

        return [success_rate,verification_rates]
    
    def set_agent_rates(self,agent_ID,success_rate, verification_rates):
        
        for x in self.rates:
            if x.agent_ID == agent_ID:
                x.success_rate = success_rate
                for i in range(0,2):
                    x.verifications_rates[i] = verification_rates[i] 
    
    def add_new_agent_rates(self, new_agent_ID, adding_agent_ID, success_rate = 0.0, verification_rates = [0.0, 0.0]):
        temp = []
        for x in self.rates:
            temp.append(x.agent_ID)
        if new_agent_ID not in temp:
            self.rates.append(Rate(new_agent_ID,success_rate,verification_rates)) 
            self.rates.sort(key=self.takeAgentID)
        else:
            print("<%s> ERROR: Agent %s rates are already in the list" %(adding_agent_ID, new_agent_ID))    
    
    def takeAgentID(self, elem): #Support method for sorting lists
        return elem.agent_ID   

class Event():

    def __init__(self, event_ID, actions_list = []): #Action("None", [ Rate("No-one", 0.0, [1.0, 1.0])]) ]):
        self.event_ID = event_ID
        self.actions_list = list(actions_list)

