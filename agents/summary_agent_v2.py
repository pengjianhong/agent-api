from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.file import FileTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.python import PythonTools
from agno.tools.mcp import MCPTools

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup

# 定义结构化输出模型
class ArticleSummary(BaseModel):
    title: str = Field(..., description="文章标题")
    one_line_summary: str = Field(..., description="一句话概述，不超过100个字")
    detailed_summary: str = Field(..., description="详细总结，不少于300字，不超过500字")
    key_points: List[str] = Field(..., description="3-5个文章核心要点，每点50-100字")
    source_url: str = Field(..., description="原文链接")

# 自定义获取微信文章的工具
class WechatFetcherTool:
    def __init__(self):
        pass
        
    def fetch_wx_article(self, url: str) -> Dict[str, Any]:
        """
        获取微信公众号文章内容，处理反爬虫机制
        
        Args:
            url: 微信公众号文章URL
            
        Returns:
            包含文章信息的字典，包括标题、公众号名称、发布时间和正文内容
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://mp.weixin.qq.com/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Upgrade-Insecure-Requests': '1',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        
        try:
            with httpx.Client(headers=headers, follow_redirects=True) as client:
                response = client.get(url, timeout=30)
                
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取文章标题
                title = soup.find('h1', class_='rich_media_title')
                title_text = title.get_text(strip=True) if title else "无法获取标题"
                
                # 提取文章作者和公众号名称
                account = soup.find('a', class_='wx_tap_link js_wx_tap_highlight weui-wa-hotarea')
                account_name = account.get_text(strip=True) if account else "未知公众号"
                
                # 提取文章正文
                content_div = soup.find('div', class_='rich_media_content')
                
                if content_div:
                    # 移除所有script标签
                    for script in content_div.find_all('script'):
                        script.decompose()
                    
                    # 获取正文内容
                    content = content_div.get_text(strip=True)
                    
                    # 提取文章发布时间
                    publish_time_div = soup.find('em', class_='rich_media_meta rich_media_meta_text')
                    publish_time = publish_time_div.get_text(strip=True) if publish_time_div else "未知时间"
                    
                    article_info = {
                        "title": title_text,
                        "account": account_name,
                        "publish_time": publish_time,
                        "content": content
                    }
                    
                    return article_info
                else:
                    # 检查是否需要验证
                    if "环境异常" in response.text or "验证" in response.text:
                        return {"error": "需要验证码验证，无法直接获取内容"}
                    return {"error": "无法找到文章内容"}
            else:
                return {"error": f"请求失败，状态码: {response.status_code}"}
        
        except Exception as e:
            return {"error": f"发生异常: {str(e)}"}

# 创建微信文章总结Agent
class WechatArticleSummarizer:
    def __init__(self):
        # 初始化微信抓取工具
        self.wechat_fetcher = WechatFetcherTool()
        
        # 初始化MCP工具，用于高级网页内容获取
        fetch_mcp_tools = MCPTools(command='uvx mcp-server-fetch')
        
        # 初始化Agent，使用DeepSeek模型
        self.agent = Agent(
            model=DeepSeek(id="deepseek-chat"),
            instructions=[
                "你是一个专业的文章总结助手，擅长分析和总结中文技术文章",
                "提取文章的核心内容，生成结构化的总结",
                "总结应包含标题、一句话概述、详细总结、关键要点和原文链接",
                "确保总结准确反映原文内容，不添加不存在的信息"
            ],
            tools=[
                fetch_mcp_tools,  # 用于高级网页内容获取
                FileTools(),      # 用于读写文件
                PythonTools(),    # 用于执行Python代码
                ReasoningTools(add_instructions=True),  # 使用推理工具分析文章结构
            ],
            response_model=ArticleSummary,  # 使用结构化输出
            markdown=True,  # 支持markdown格式输出
        )
        
    def fetch_and_summarize(self, url):
        """获取并总结微信文章"""
        # 首先使用自定义工具获取文章内容
        article_data = self.wechat_fetcher.fetch_wx_article(url)
        
        if "error" in article_data:
            print(f"获取文章失败: {article_data['error']}")
            return None
        
        # 将文章内容保存到临时文件
        with open("temp_article.txt", "w", encoding="utf-8") as f:
            f.write(f"标题: {article_data['title']}\n")
            f.write(f"公众号: {article_data['account']}\n")
            f.write(f"发布时间: {article_data['publish_time']}\n\n")
            f.write(article_data['content'])
        
        # 构建提示词，指向保存的文件而不是URL
        prompt = f"""
        请总结以下微信公众号文章，文章内容已保存在temp_article.txt文件中。
        
        文章标题: {article_data['title']}
        公众号: {article_data['account']}
        
        步骤:
        1. 阅读文章内容
        2. 分析文章的主题和结构
        3. 用一句话概括文章主要内容(不超过100字)
        4. 撰写详细总结(300-500字)
        5. 提炼3-5个核心要点(每点50-100字)
        6. 以markdown格式输出结果
        
        原文链接: {url}
        """
        
        # 调用agent获取结果
        response = self.agent.run(prompt)
        
        # 将结果保存到文件
        self.save_summary_to_file(response.content)
        
        return response.content
    
    def save_summary_to_file(self, summary):
        """将总结保存为markdown文件"""
        content = f"""# {summary.title}
        ## 一句话概述
        {summary.one_line_summary}
        ## 总结
        {summary.detailed_summary}
        ## 重点划线
        """
        
        # 添加关键点
        for i, point in enumerate(summary.key_points, 1):
            content += f"{i}. {point}\n"
            
        content += f"\n## 原文链接\n{summary.source_url}"
        
        # 保存到文件
        with open("article_summary.md", "w", encoding="utf-8") as f:
            f.write(content)

# 使用示例
if __name__ == "__main__":
    summarizer = WechatArticleSummarizer()
    # url = "https://mp.weixin.qq.com/s/GrYISaYsvcMNtaghLZfqxQ"
    url = "https://mp.weixin.qq.com/s/GqsD8byer_8GPIBY9Njhag"
    summary = summarizer.fetch_and_summarize(url)
    if summary:
        print(f"文章《{summary.title}》总结已生成并保存到article_summary.md")
    else:
        print("文章获取失败，无法生成总结")