@echo off
REM This batch script updates the remote repository to match your local repository

REM Get current branch name and store it in a variable
for /f "tokens=*" %%i in ('git branch --show-current') do set branch=%%i
echo Current branch is: %branch%

REM Stage all changes (including modifications and deletions)
git add -A

REM Prompt for a commit message
set /p commitMessage="Enter commit message: "

REM Commit the changes
git commit -m "%commitMessage%"

REM Push the changes to the remote repository
git push origin %branch%

echo Remote repository updated.
pause