import pandas as pd
import random
from openai import AzureOpenAI

class PoliticalLLM:
    def __init__(self, vm_file, candidates, api_key, endpoint, deployment_name):
        self.vm = pd.read_csv(vm_file)
        self.candidates = candidates
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name

        # Initialize OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint="https://jiahu-mcviy2ug-eastus2.cognitiveservices.azure.com/",
            api_version="2024-12-01-preview"
        )
    
    def load_state(self, state_file):
        """Load a new state CSV file and prepare grouping."""
        self.df = pd.read_csv(state_file)
        self.desc_cols = []

        for column in self.df.columns:
            vm_subset = self.vm[self.vm['var_id'] == column]
            if not vm_subset.empty:
                value_dict = dict(zip(vm_subset['value_id'], vm_subset['var_values']))
                description = vm_subset['description'].iloc[0].upper()
                self.df[description] = self.df[column].map(value_dict)
                self.desc_cols.append(description)
        
        self.group_keys = self.desc_cols + ['loc_msa']
        self.grouped = self.df.groupby(self.group_keys)
    
    def build_prompt(self, sample_row, state_name):
        demographic_info = ', '.join([str(sample_row[col]) for col in self.desc_cols])
        loc_desc = 'Small town' if sample_row['loc_msa'] == 'S' else 'Large city'

        prompt = f"If the person's info is: {demographic_info}, lives in {loc_desc}, in {state_name}\n"
        prompt += "Here is the list of candidates:\n"
        for c in self.candidates:
            prompt += f"- {c}\n"
        prompt += "Based on this person's info, what is the probability that this person votes for each candidate?\n"
        prompt += "Please give your answer in this clean format (each on a new line, and no other information):\n"
        for c in self.candidates:
            prompt += f"{c}: xx%\n"

        return prompt

    def call_LLM(self, prompt):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful prediction assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    
    def poll(self, state_name):
        totals = {c: 0 for c in self.candidates}  # Initialize totals per candidate

        for group_key, group_df in self.grouped:
            print(f"Processing group {group_key} in state {state_name}...")

            sample_row = group_df.sample(n=1).iloc[0]
            prompt = self.build_prompt(sample_row, state_name)
            prob = self.call_LLM(prompt)
            print(f"Got prediction: {prob}") 

            # Get total population for this group
            group_population = len(group_df)

            # Process prob text to extract candidate probabilities
            for line in prob.split('\n'):
                line = line.strip()
                if ':' in line:
                    candidate, perc_str = line.split(':')
                    candidate = candidate.strip()
                    perc_str = perc_str.strip().replace('%','')

                    if candidate in totals and perc_str.replace('.','',1).isdigit():
                        percentage = float(perc_str)
                        num_voters = (percentage / 100) * group_population
                        totals[candidate] += num_voters

        return totals

