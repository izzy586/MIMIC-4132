 #!/bin/bash

    # Navigate to your local repository directory
    cd /path/to/your/local/repo

    # Add all changes to the staging area
    git add .

    # Commit changes with a timestamp
    git commit -m "Automated backup: $(date +'%Y-%m-%d %H:%M:%S')"

    # Push changes to GitHub
    git push origin main # or 'master' depending on your branch name