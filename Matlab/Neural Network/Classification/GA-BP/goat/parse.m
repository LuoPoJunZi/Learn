function x = parse(inStr)
% parse - 解析由空格分隔的字符串向量，转换为字符串矩阵
%
% 输入参数:
%   inStr - 由空格分隔的字符串向量
%
% 输出参数:
%   x     - 解析后的字符串矩阵，每行对应一个子字符串

    %% 切割字符串
    strLen = size(inStr, 2); % 输入字符串的长度
    x = blanks(strLen);      % 初始化输出矩阵
    wordCount = 1;           % 初始化单词计数
    last = 0;                % 初始化上一个空格的位置
    for i = 1 : strLen
        if inStr(i) == ' '      % 如果当前字符为空格
            wordCount = wordCount + 1;            % 增加单词计数
            x(wordCount, :) = blanks(strLen);      % 初始化下一行
            last = i;                              % 更新上一个空格的位置
        else
            x(wordCount, i - last) = inStr(i);    % 填充当前单词的字符
        end
    end
end
