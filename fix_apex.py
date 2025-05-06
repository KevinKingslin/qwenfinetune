import sys

def modify_line(file_path, line_number, new_content):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        lines[line_number - 1] = new_content + '\n'
        
        with open(file_path, 'w') as file:
            file.writelines(lines)
            
    except Exception as e:
        print(f"An error occured: {e}")
        
    print("Fixed apex")
        
if __name__ == "__main__":
    file_path = "/usr/local/lib/python3.10/dist-packages/apex/amp/_initialize.py"
    line_number = 2
    new_content = "string_classes = str"
    modify_line(file_path, line_number, new_content)