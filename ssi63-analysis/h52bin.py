import struct
import pandas as pd

df = pd.read_hdf('ssi63.h5',start=20*60*60*100, stop=20*60*60*120)

vrs = ['SPAG_FREQ', 'SPAG_K', 'SPAG_B_DLDT', 'SPAG_V_DLDT', 'SPAG_RATE_MIN', 'SPAG_RATE_MAX', 'SPAG_B_TMIN', 'SPAG_V_TMIN', 'SPAG_H_CMD', 'ALTITUDE_BAROMETER']
typ = ['f' for _ in range(len(vrs))]


vrs.extend(['SPAG_EFFORT', 'SPAG_VENT_TIME_INTERVAL', 'SPAG_BALLAST_TIME_INTERVAL', 'SPAG_VALVE_INTERVAL_COUNTER', 'SPAG_BALLAST_INTERVAL_COUNTER', 'ACTION_SPAG', 'SPAG_VENT_TIME_TOTAL', 'SPAG_BALLAST_TIME_TOTAL'])
vrs = [vr.lower() for vr in vrs]
typ.extend(['f','f','f','f','I','I','I', 'I'])

m = {'f': 'float', 'i': 'int', 'I': 'unsigned int'}

open("header.h","w").write("typedef struct __attribute__ ((packed)) {\n"+'\n'.join(["  "+m[t] +" "+v.upper()+";" for t, v in zip(typ, vrs)])+"\n} miniframe;\n")

S = '<' + ''.join(typ)

f = open("data.bin","wb")

for i in range(len(df)):
    vals = [df[vr][i] for vr in vrs]
    print(df['spag_effort'][i])
    print(vals)
    print(S)
    d = struct.pack(S, *vals)
    #print len(d)
    f.write()

f.close()