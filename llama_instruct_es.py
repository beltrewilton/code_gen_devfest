import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    GenerationConfig,
)

model_id = "clibrain/Llama-2-ft-instruct-es"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_id)


def create_instruction(instruction, input_data=None, context=None):
    sections = {
        "Instrucción": instruction,
        "Entrada": input_data,
        "Contexto": context,
    }

    system_prompt = "A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n"
    prompt = system_prompt

    for title, content in sections.items():
        if content is not None:
            prompt += f"### {title}:\n{content}\n\n"

    prompt += "### Respuesta:\n"

    return prompt


def generate(
    instruction,
    input=None,
    context=None,
    max_new_tokens=128,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    **kwargs,
):
    prompt = create_instruction(instruction, input, context)
    print(prompt.replace("### Respuesta:\n", ""))
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    attention_mask = inputs["attention_mask"].to("mps")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Respuesta:")[1].lstrip("\n")


instruction = "Dame una lista de lugares a visitar en República Dominicana."
print(generate(instruction))
