import argparse
import ast


def parse_list_arg(s):
    """
    解析命令行参数中的列表字符串，将其转换为Python列表对象

    该函数使用 ast.literal_eval 安全地解析字符串字面量，支持列表、元组、字典等格式，
    但只接受列表类型，其他类型会抛出异常。

    Args:
        s (str): 要解析的字符串，格式应为有效的Python列表字面量，
                 例如: '["item1", "item2"]' 或 '[1, 2, 3]'

    Returns:
        list: 解析后的列表对象

    Raises:
        argparse.ArgumentTypeError: 当输入格式无效或不是列表类型时抛出

    Example:
        >>> parse_list_arg('["a", "b", "c"]')
        ['a', 'b', 'c']
        >>> parse_list_arg('[1, 2, 3]')
        [1, 2, 3]
    """
    try:
        result = ast.literal_eval(s)

        # 确保解析结果是列表类型
        if not isinstance(result, list):
            raise ValueError("输入不是 list 类型")

        return result
    except (ValueError, SyntaxError) as e:
        # 当解析失败时，提供友好的错误信息
        raise argparse.ArgumentTypeError(f'无效的 list 格式: {s}，应为类似 \'["1", "2"]\' 的字符串。错误: {e}')
