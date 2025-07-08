from political_llm.py import PoliticalLLM

# Initialize model
my_obj = PoliticalLLM(state_file="AK.csv",
                      vm_file="variable_mapping.csv",
                      candidates=["Joe Biden", "Donald Trump", "Bill Clinton"])

# Get predicted results
results = my_obj.poll(state_name="Alaska")
