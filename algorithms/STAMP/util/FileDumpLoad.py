import dill as cp


def dump_file(*dps):
    '''
    dump file.
    dps: [data, path]s.
    '''
    for dp in dps:
        if len(dp) != 2:
            print("issue:" + str(dp))
            continue
        dfile = open(dp[1],'wb')
        cp.dump(dp[0], dfile)
        dfile.close()
    print ("dump file done.")


def load_file(*ps):
    '''
    load file.
    ps: [path,...]s
    '''
    ret = []
    for p in ps:
        dfile = open(p, 'rb')
        ret.append(cp.load(dfile))
    return ret
