from retriving import retrieve_relevant_resources

def prompt_formatter(query: str,
                     context_items: list[dict],
                     tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.

    # base_prompt = """Based on the following context items, please answer the query.
    #                 Give yourself room to think by extracting relevant passages from the context before answering the query.
    #                 Don't return the thinking, only return the answer.
    #                 Make sure your answers are as explanatory as possible.
    #                 Use the following examples as reference for the ideal answer style.
    #                 \nExample 1:
    #                 Query: What are the fat-soluble vitamins?
    #                 Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    #                 \nExample 2:
    #                 Query: What are the causes of type 2 diabetes?
    #                 Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    #                 \nExample 3:
    #                 Query: What is the importance of hydration for physical performance?
    #                 Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
    #                 \nNow use the following context items to answer the user query:
    #                 {context}
    #                 \nRelevant passages: <extract relevant passages from the context here>
    #                 User query: {query}
    #                 Answer:"""
    
    base_prompt = """Based on the following context items, please answer the query.
                    Take your time to think by extracting relevant passages from the context before answering the query.
                    Only return the final answer, not the thought process.
                    Ensure your answers are as detailed and magical as possible.
                    Use the following examples as reference for the ideal answer style.
                    \nExample 1:
                    Query: What are the core materials used in wand making?
                    Answer: The core materials used in wand making include Phoenix feather, Dragon heartstring, and Unicorn hair. These cores are known for their powerful magical properties. Phoenix feathers are rare and produce wands with a great range of magic. Dragon heartstrings create wands with immense power and are prone to accidents. Unicorn hair wands are the most consistent, producing the most reliable and faithful magic.
                    \nExample 2:
                    Query: What are the primary methods of transportation in the wizarding world?
                    Answer: The primary methods of transportation in the wizarding world include Apparition, Floo Powder, Portkeys, and broomsticks. Apparition allows wizards to teleport instantly from one location to another but requires practice to avoid splinching. Floo Powder enables travel via the Floo Network, connecting fireplaces for quick travel. Portkeys are enchanted objects that transport wizards to a pre-determined location at a specific time. Broomsticks are used for both travel and sport, such as in Quidditch, providing a versatile and iconic mode of transport.
                    \nExample 3:
                    Query: How does the Time-Turner work and what are its limitations?
                    Answer: The Time-Turner is a magical device that allows the user to travel back in time. It consists of a small hourglass on a necklace, which when turned, takes the wearer back one hour for each rotation. However, the Time-Turner has strict limitations to prevent catastrophic changes to the timeline. Users must avoid being seen by their past selves and others to prevent paradoxes. Prolonged use can cause significant strain and potentially irreversible alterations to events, making it a tool that must be used with extreme caution.
                    Now use the following context items to answer the user query:
                    {context}
                    Relevant passages: <extract relevant passages from the context here>
                    User query: {query}
                    Answer:"""


    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def ask_llm(query,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True,
        return_answer_only=True,
        tokenizer=None,
        llm_model=None,
        embeddings=None,
        pages_and_chunks=None,
        embedding_model=None
        ):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  embedding_model=embedding_model,
                                                  n_resources_to_return=3)

    print(scores, indices)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")#.to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    return output_text, context_items


def ask_llm_streaming(query,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True,
        return_answer_only=True,
        tokenizer=None,
        llm_model=None,
        embeddings=None,
        pages_and_chunks=None,
        embedding_model=None
        ):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  embedding_model=embedding_model,
                                                  n_resources_to_return=3)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    if return_answer_only:
        # Only return the answer without the context items
        yield output_text
    else:
        # Return both answer and context items
        yield output_text, context_items

        # Stream the decoded tokens
        decoded_tokens = tokenizer.convert_ids_to_tokens(outputs[0])
        for token in decoded_tokens:
            yield token

