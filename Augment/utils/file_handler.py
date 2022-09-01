def write_file(file_name, context):
    f = open(file_name, 'w', encoding='utf-8')
    f.write(str(context))
    f.close()
    print(f'Finish writing {file_name} !')

def read_file(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    content = f.read()
    return eval(content)
