

def convert_files_list_and_query_to_messages(results, query):
    messages = []
    messages.append(
        {"role": "system", "content": "You are a helpful code assistant"})
    content_start = f"I want you to help me with a code question in a retrival augmented code search system. I'm going to give you the top {
        len(results)} that are related to the question, and the question itself, and would like your ideas regarding how to solve it."
    files_str = ''
    for result in results:
        file_name, content, score = result
        files_str += file_name + ':\n'
        files_str += content + '\n'
    full_content = content_start + files_str + 'QUESTION:\n' + query
    messages.append({"role": "user", "content": full_content})
    return messages
