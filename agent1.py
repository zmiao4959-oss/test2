"""
只是一个独立的agent模块，不适合被import
"""

import ast
import inspect
import os
import platform
import re
from string import Template
from typing import Callable, List, Tuple

# import click
from dotenv import load_dotenv
from lammps import run_lammps_with_monitor
from openai import OpenAI
from prompt_template import react_system_prompt_template


class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        # <question>content</question>  -><question>与</question>:XML/HTML风格标签的开始标签和结束标签
        messages = [
            {
                "role": "system",
                "content": self.render_system_prompt(react_system_prompt_template),
            },
            {"role": "user", "content": f"<question>{user_input}</question>"},
        ]

        while True:
            # 请求模型
            content = self.call_model(messages)

            # 检测 Thought
            """
                r":原始字符串（Raw String），如print(r"/n")输出为"/n"而不是换行。f":格式化字符串（Formatted String）格式化字符串允许在字符串中嵌入表达式
                <thought> 匹配开始标签
                (.*?) 是一个捕获组，匹配任意字符（包括换行符，因为使用了re.DOTALL）非贪婪模式，即匹配到第一个</thought>就结束
                </thought> 匹配结束标签
                re.DOTALL 标志使得 . 匹配包括换行符在内的任意字符
            """
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                """
                content="我喜欢去山上玩"
                a = re.search(r"我(.*?)去(.*?)玩", content, re.DOTALL)
                print(a.group(0))#我喜欢去山上玩
                print(a.group(1))#喜欢
                print(a.group(2))#山上
                """
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(
                    r"<final_answer>(.*?)</final_answer>", content, re.DOTALL
                )
                # return出run函数
                return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")

            """
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"
            """

            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            # time.sleep(300)
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            # inspect.signature(func) -> 输出：(函数的参数)
            signature = str(inspect.signature(func))
            # inspect.getdoc(func)->输出函数或者类里面的第一个注释，只能是'''注释且注释只能从第一行就开始
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        # os.listdir(self.project_directory)->输出目标目录里面所有文件的名字，包括文件夹和各种文件(.txt等)，输出到一个列表里面
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        # Template(string).substitute(a=a1,b=b1)-> 把string里面的${a}与${b}替换成a1与b1
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list,
        )

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("api_key")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key

    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content

        # print("我想要的内容在下面：\n",content)
        # time.sleep(300)

        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        # 在re.match()、re.search()等函数里面，即使是r""也是按照正则表达式判定而不是按照r""的原始字符串表达，因此"\("就是"("的意思
        # (\w+):匹配字母数字下划线 因此对于"f_x1\n("here")"这样的字符串，(\w+)只能匹配到"f_x1"，而"\("匹配"(",也就是说"\n"无法匹配，因此返回None

        match = re.match(r"(\w+)\((.*)\)", code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")
        # print("这里是我要看的命令：\n")
        # print(code_str)
        func_name = match.group(1)
        # .strip()->从字符串的首与尾开始检测空格或者\n\t等并删除，检测失败即停止
        args_str = match.group(2).strip()
        # print(match.group(2))
        # print(func_name,"\n",args_str)
        # time.sleep(300)
        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == "(":
                    paren_depth += 1
                    current_arg += char
                elif char == ")":
                    paren_depth -= 1
                    current_arg += char
                elif char == "," and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    # print(current_arg.strip())
                    # time.sleep(300)
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i - 1] != "\\"):
                    in_string = False
                    string_char = None

            i += 1

        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or (
            arg_str.startswith("'") and arg_str.endswith("'")
        ):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace("\\n", "\n").replace("\\t", "\t")
            inner_str = inner_str.replace("\\r", "\r").replace("\\\\", "\\")
            return inner_str

        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            # ast.literal_eval->类似于eval，但是只把字符串转化为对象比如"[1,2]"变成一个list:[1,2]
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        os_map = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}
        """
        1.platform.system()->依据左边的系统：
            # macOS:    "Darwin"
            # Windows:  "Windows"
            # Linux:    "Linux"
        返回右边的字符串
        2.dict.get(key, default) 如果dict里有key则返回key的value，否则返回default
        """
        return os_map.get(platform.system(), "Unknown")


def read_file(file_path):
    """用于读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess

    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr


def run_lammps(file_path, lammps_file):
    """
    运行 LAMMPS 并监控运行状态

    Args:
        file_path: LAMMPS 文件所在目录
        lammps_file: LAMMPS 输入文件名
    """
    result = run_lammps_with_monitor(file_path, lammps_file)
    return result


def run_ovito(file_path, ovito_file):
    """
    运行 LAMMPS 并监控运行状态

    Args:
        file_path: OVITO 文件所在目录
        ovito_file: OVITO 输入文件名
    """
    import os
    import subprocess

    # 定义参数
    input_file = os.path.join(file_path, ovito_file)
    input_file = os.path.normpath(input_file)
    # 找到可用的ovito路径
    ovito_exe = r"D:\1_app\OVITO\OVITO Basic\ovito.exe"

    if ovito_exe is None:
        print("未找到ovito.exe，请手动指定路径")
        # 或者让用户输入路径
        ovito_exe = input("请输入ovito.exe的完整路径: ")
    else:
        print(f"找到ovito: {ovito_exe}")

    # 调用ovito并传递参数
    result = subprocess.run(
        [
            ovito_exe,
            input_file,
        ],
        capture_output=True,
        text=True,
    )

    print("返回码:", result.returncode)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        return f"ovito打开失败，失败原因为{result.stdout}"
    return "ovito运行成功！"


"""
@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))"""


def main(project_directory):
    project_dir = os.path.abspath(project_directory)
    tools = [read_file, write_to_file, run_terminal_command, run_lammps, run_ovito]
    agent = ReActAgent(
        tools=tools, model="deepseek-chat", project_directory=project_dir
    )

    task = input("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    # project_path = "C:/Users/LENOVO/OneDrive/Desktop/Agent/VideoCode-main/Agent的概念、原理与构建模式"
    # main(r"D:\a桌边文件\Agent\VideoCode-main\Agent的概念、原理与构建模式")
    main(os.getcwd())
