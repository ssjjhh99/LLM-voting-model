import os
from political_llm import PoliticalLLM

if __name__ == '__main__':
    vm_file = "variable_mapping.csv"
    api_key = input("Enter your Azure OpenAI API key: ")
    endpoint = input("Enter your Azure endpoint URL: ")
    deployment_name = input("Enter your deployment name: ")
    candidates = input("Enter all candidate names separated by commas: ").split(",")
    candidates = [c.strip() for c in candidates]
    vote_counts = {candidate: 0 for candidate in candidates}


    # Initialize model
    poller = PoliticalLLM(vm_file, candidates, api_key, endpoint, deployment_name)

    # Loop through each CSV in data folder
    data_folder = "data"
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            state_path = os.path.join(data_folder, filename)
            state_name = filename.replace(".csv", "")


            # Load this state's data
            poller.load_state(state_path)

            # Get predicted results
            results = poller.poll(state_name)

            for candidate, number in results.items():
                vote_counts[candidate] += number

    winner = max(vote_counts, key=vote_counts.get)
    print(f"The winner is {winner} with {vote_counts[winner]} votes.")
