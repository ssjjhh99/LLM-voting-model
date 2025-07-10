import os
import multiprocessing
from political_llm import PoliticalLLM


def process_state(args):
    state_file, vm_file, candidates, api_key, endpoint, deployment_name = args

    # Initialize poller object inside each process
    poller = PoliticalLLM(vm_file, candidates, api_key, endpoint, deployment_name)

    state_name = os.path.basename(state_file).replace(".csv", "")
    poller.load_state(state_file)
    results = poller.poll(state_name)
    
    # Return result dictionary
    return results


if __name__ == "__main__":
    import time

    vm_file = "variable_mapping.csv"
    api_key = input("Enter your Azure OpenAI API key: ")
    endpoint = input("Enter your Azure endpoint URL: ")
    deployment_name = input("Enter your deployment name: ")
    candidates = input("Enter all candidate names separated by commas: ").split(",")

    start_time = time.time()

    data_folder = "data"
    state_files = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            state_path = os.path.join(data_folder, filename)
            state_files.append(state_path)

    # Prepare args list for each process
    args_list = [(state_file, vm_file, candidates, api_key, endpoint, deployment_name)
                 for state_file in state_files]

    # Initialize vote_counts dictionary
    vote_counts = {c.strip(): 0 for c in candidates}

    # Use multiprocessing pool
    with multiprocessing.Pool(processes=min(len(args_list), multiprocessing.cpu_count())) as pool:
        results_list = pool.map(process_state, args_list)

    # Combine results
    for results in results_list:
        for candidate, number in results.items():
            vote_counts[candidate.strip()] += number

    # Print final winner
    winner = max(vote_counts, key=vote_counts.get)
    print(f"The winner is {winner} with {vote_counts[winner]} votes.")

    print(f"Finished in {time.time() - start_time:.2f} seconds")
