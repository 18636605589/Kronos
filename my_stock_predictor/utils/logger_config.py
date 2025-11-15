#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志配置模块
提供统一的日志配置和格式化输出
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = 'stock_predictor',
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回配置好的logger
    
    Args:
        name: logger名称
        level: 日志级别
        format_string: 自定义格式字符串
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        log_file: 日志文件路径
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 控制台handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件handler
    if enable_file and log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'stock_predictor') -> logging.Logger:
    """
    获取已配置的logger，如果不存在则创建
    
    Args:
        name: logger名称
    
    Returns:
        logger实例
    """
    logger = logging.getLogger(name)
    
    # 如果logger还没有handler，进行基本配置
    if not logger.handlers:
        setup_logger(name)
    
    return logger


# 便捷函数：用于用户友好的输出（保留emoji和格式化）
def user_print(message: str, emoji: str = ''):
    """
    用户友好的打印函数，保留emoji和格式化
    
    注意：此函数用于关键用户提示，不是所有print都应该替换
    """
    if emoji:
        print(f"{emoji} {message}")
    else:
        print(message)

