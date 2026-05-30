import tinker

service_client = tinker.ServiceClient()
sampling_path = "tinker://0eff6dbd-6564-57df-90f8-a1aacb6c7616:train:0/sampler_weights/000020"
sampling_client = service_client.create_sampling_client(model_path=sampling_path)

prompt = tinker.ModelInput.from_string(r"What is 24 * 15 + 37? Provide a numerical answer without units, written inside \boxed{}.")

result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params={"temperature": 0.8, "top_p": 0.95}).result()
for seq in result.sequences:
    print(seq.token_ids)
