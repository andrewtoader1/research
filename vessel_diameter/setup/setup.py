import os

#upgrade pip to latest version
os.system("python3 -m pip install --upgrade pip")


#install all modules needed to run script
os.system("python3 -m pip install nd2reader")
os.system("python3 -m pip install numpy")
os.system("python3 -m pip install numba")
os.system("python3 -m pip install scipy")
os.system("python3 -m pip install matplotlib")
os.system("python3 -m pip install xlsxwriter")
os.system("python3 -m pip install opencv-python")

#clear command screen
os.system("clear")