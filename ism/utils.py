def format_seq_file(seq_file_path):
    seq_file = open(seq_file_path, "r")
    contents = seq_file.read()
    seq_file.close()

    contents = contents.split(',')

    new_contents = []

    for content in contents:
        stop_idx = content.find(']')
        new_contents.append(content[:stop_idx + 1])

    filtered_contents = list(filter(lambda x: len(x) != 0, new_contents))

    filtered_contents = [int(content[1:-1]) for content in filtered_contents]

    filtered_contents = [filtered_contents]

    return filtered_contents