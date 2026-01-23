import adi 

sdr = adi.Pluto('ip:192.168.0.150') # or whatever your Pluto's IP is
sdr.sample_rate = int(2.5e6)
sdr.rx()