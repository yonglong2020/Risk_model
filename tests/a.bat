@echo off
:: 设置中文乱码修复
chcp 65001 >nul
echo 正在执行准备工作，请稍候...

:: --- 第一部分：打开 Revit ---
echo 2. 正在启动 Revit 软件...
:: 请将下面的路径替换为你刚才在属性里复制的真实路径
start "" "C:\Program Files\Autodesk\Revit 2016\Revit.exe"

:: --- 第二部分：打开 U 盘里的便携版浏览器 ---
echo 3. 正在启动 Supermium 浏览器...
:: 假设你的 Supermium 启动程序就在 U 盘根目录，文件夹叫 SupermiumPortable
start "" "%~dp0SupermiumPortable\SupermiumPortable.exe"

:: --- 第三部分：拷贝课件 ---
:: %~dp0 代表脚本所在的文件夹路径，能自动识别U盘盘符
echo 1. 正在拷贝课件文件夹到桌面...
xcopy "%~dp0B【备课】BIM" "%USERPROFILE%\Desktop\B【备课】BIM\" /E /I /Y /Q


echo ---------------------------------------
echo 准备就绪！课件已拷贝至桌面，软件已启动。
echo 祝老师上课顺利！
echo ---------------------------------------
timeout /t 5