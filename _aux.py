############# auxiliary functions for limiter in 1D/2D(trigs)

import numpy as np
from ngsolve import VOL, GridFunction, L2, grad, SEGM, IntegrationRule


######### PART 1: GEOMETRIC INFORMATION 1D & 2D

### getVerts1D & getVerts2D (trigs only) return ALL the pplimiter evaluation points
def getVerts1D(mesh, ir=[1,0]):
    mpts = mesh([0],[0], [0])
    mpt = mpts[0][3]
    # generate all integration pts
    npts = len(ir)
    mips = np.zeros((npts*mesh.ne,), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), 
                ('meshptr', '<f8'), ('VorB', '<i4'), ('nr', '<i4')])
    for i in range(mesh.ne):
        for j in range(npts):
            mips[npts*i+j] = (ir[j], 0, 0, mpt, 0, i)
    return mips

def getVerts2D(mesh, order):
    mpts = mesh([0],[0], [0]) 
    mpt = mpts[0][3]
    nd = int((order+1)*(order+2)/2)
    # generate all gauss nodes on the edges   
    ir = []       
    gaussrule = IntegrationRule(SEGM, 2*order+1)
    for i in range(order+1):                                                    
        ir.append((gaussrule.points[i][0],0)) # bottom edge                     
        ir.append((gaussrule.points[i][0],1-gaussrule.points[i][0])) # long edge
        ir.append((0,gaussrule.points[i][0])) # right edge                      
    npts = len(ir)                                                              
    mips = np.zeros((npts*mesh.ne,), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                                           ('meshptr', '<f8'), ('VorB', '<i4'), ('nr', '<i4')])            
    for i in range(mesh.ne):                                                     
        for j in range(npts):                                                   
            mips[npts*i+j] = (ir[j][0], ir[j][1], 0, mpt, 0, i)     
    return mips

# getNeighbors1D/2D return all ngbr geometric info needed for the TVB limiter && FuShu indicator
def getNeighbors1D(mesh, bc="periodic"):
    listNbs = ([0 for i in range(mesh.ne)] , [0 for i in range(mesh.ne)]) 
    for el in mesh.Elements(VOL):
        nr = el.nr
        vertices = el.vertices
        for i in range(len(vertices)):
            eb = mesh[vertices[i]].edges
            # FIXME LATER a boundary edge (use current cell)
            if len(eb) == 1: 
                listNbs[i][nr] = nr
            else:
                nr1, nr2 = eb[0].nr, eb[1].nr
                if nr1==nr:
                    listNbs[i][nr] = nr2 
                else :
                    listNbs[i][nr] = nr1

    ### FIXME manual fix for periodic boundary nodes
    if bc=="periodic":
        if listNbs[0][0] ==0 : listNbs[0][0]=mesh.ne-1
        if listNbs[1][-1] == mesh.ne-1 : listNbs[1][-1]=0
    return listNbs


# You need an afternoon to digest this function
def getNeighbors2D(mesh, eps0=1e-14):
    # 1: get barycenter coordinates: dict
    baryC = {}
    for el in mesh.Elements():        
        verts = el.vertices
        coo = [0, 0]
        for vert in verts:
            vc = mesh[vert].point
            coo[0] += vc[0]
            coo[1] += vc[1]
        coo[0] /=3
        coo[1] /=3            
        baryC[el.nr] = coo
    
    # 2: get neighbor cell numbers & alpha values from coo
    listNbr = []
    listNbj = []
    listNbw = []
    alphaList = []
    listNrms = []
    for el in mesh.Elements():        
        nr = el.nr
        # the barycenters
        b0 = np.array(baryC[nr])
        
        edges = el.edges
        # HACK orientation
        verts = el.vertices
        if verts[1].nr > verts[2].nr:
            edges = [edges[2], edges[1], edges[0]]
            verts = [verts[0], verts[2], verts[1]]
            
        # The 3x3 mat0
        mat0 = []
        for i in range(3):
            cx = mesh[verts[i]].point
            mat0.append([1, cx[0],cx[1]])
        imat0 = np.linalg.inv(mat0).T
        
        
        nbr = [] # nbr cell number (to obtain cell average)
        nbw = [] # nbr cell weights for computing extended cell averages
        nbj = [] # nbr cell loc
        bs, ms = [], [] # coordinates
        bcs = []
        nrms = []
        for ii, edge in enumerate(edges):
            # compute edge midpoint coo - barycenter
            vts = mesh[edge].vertices
            coo = [0,0]
            for vt in vts:
                vc = mesh[vt].point
                coo[0] += vc[0]
                coo[1] += vc[1]
            coo[0] /= 2
            coo[1] /= 2
            coo = np.array(coo) - b0
            ms.append(coo)
            
            ### compute normal direction!
            nrm = np.array(mesh[vts[0]].point)-np.array(mesh[vts[1]].point)
            nrm0 = nrm/np.linalg.norm(nrm)
            nrms.append(nrm0)
            
            # get neighbor cell info
            nbs = mesh[edge].elements
            if len(nbs)==1: # a boundary edge, set coord == 0
                bcs.append(ii)
                nbr.append(-1)
                nbj.append(-1) 
                nbw.append(-1)
            else:
                for nb in nbs:
                    if nb.nr != nr:
                        nbr.append(nb.nr)
                        bj = baryC[nb.nr]
                        coo = np.array(bj)-b0
                        bs.append(coo)
                        
                        # determin extended avg loc
                        vx = mesh[nb].vertices
                        # determine if nbr is a boundary cell ??
                        
                        v2 = vx[0].nr+vx[1].nr+vx[2].nr-vts[0].nr-vts[1].nr
                        if v2 > vts[1].nr:
                            nbj.append(2)
                        elif v2 > vts[0].nr:
                            nbj.append(0)
                        else:
                            nbj.append(1)
                        
                        # extended weights
                        bw = imat0.dot(np.array([1, bj[0], bj[1]]))
                        nbw.append(bw)
        listNbr.append(nbr)
        listNbw.append(nbw)
        listNbj.append(nbj)
        listNrms.append(nrms)
        
        # compute data related to interior triangles
        if len(bs)==3: # a interior trig
            # the three 2x2 matrices -> 3x4 
            #(i) 1-2/2-3/3-1
            mats = []
            for j in range(3):
                k = (j+1)%3
                # the inverse matrix
                mats.append(np.linalg.inv(np.array([bs[j], bs[k]])).T)
            # loop over edges
            alphas = []
            
            # loop over edges
            ids = [[0, 2], [0, 1], [1,2]]
            for j in range(3):
                alpha0 = mats[ids[j][0]].dot(ms[j])
                i0 = ids[j][0]
                if alpha0[0]< -eps0 or alpha0[1]<-eps0: # negative, skip to next pair
                    alpha0 = mats[ids[j][1]].dot(ms[j])
                    i0 = ids[j][1]    
                alphas.append([alpha0[0], alpha0[1], i0])                
        elif len(bs)==2: # boundary cell: with two nbrys
            ii = (bcs[0]+1)%3
            mat = np.linalg.inv(np.array([bs[0], bs[1]])).T
            # loop over edges
            alphas = []
            
            # loop over edges
            ids = [[0, 2], [0, 1], [1,2]]
            for j in range(3):
                alpha0 = mat.dot(ms[j])
                if alpha0[0]< -eps0 or alpha0[1]<-eps0: # negative, no limit in this direction
                    alphas.append([0, 0, -1])                
                else:
                    alphas.append([alpha0[0], alpha0[1], ii])                
        elif len(bs)==1: # corner cell: NO LIMIT
            # loop over edges
            alphas = []            
            for j in range(3):
                alphas.append([0,0, -1])
        alphaList.append(alphas)
    return listNbr, listNbj, listNbw, alphaList, listNrms

    
######### PART 2: FuShu indicator (with a new scaling denominator) implemenation 
# reference: [G. Fu and C.-W. Shu, JCP, vol 347, pp 305-327]
# id = \sum_{nbr} |p_avg_ext(nbr) - p_avg|/|pmax-pmin|
# The new scaling is suggested by Prof. Shu and it works for small perturbation test for SW 
# The old scaling is local |p_max|, which does not work for small perturbation test.

def computeIndicator1D(gf, k, listNbs, flag="du"): # k: polynomial degree
    mesh = gf.space.mesh
    nelems = mesh.ne
    u0 = gf.vec[0:nelems].FV().NumPy()
    # higher order terms
    view = gf.vec[mesh.ne:].FV().NumPy()
    u1 = view[::k] # P1 part
    uL = u0-2*u1
    uR = u0+2*u1
    # extrapolated cell averages @ current cell
    uL0=uR[listNbs[0]] 
    uR0=uL[listNbs[1]] 
    
    if flag=="du": # Shu denominator
        diff = u0.max() - u0.min()
        ind = (abs(u0-uR0)+abs(u0-uL0))/(diff+1e-8*u0.max())
    else: # not implemented
        print("Please provide a good scaling for Indicator1D")
        stop
    # HACK: no limiting for LEFT/RIGHT NODES (limiting boundary cell may cause issue...)
    ind[0] = 0
    ind[-1] = 0
    return ind

def computeIndicator2D(gfu, order, mesh, listNbr, listNbj, listNbw, flag="du"):
    # cell averages
    u0 = gfu.vec[:mesh.ne].FV().NumPy()
    hdof = int((order+1)*(order+2)/2-1)
    # cell averages
    uho = gfu.vec[mesh.ne:].FV().NumPy().reshape(mesh.ne, hdof)
    # P1 parts
    u1 = uho[:,0]
    u2 = uho[:,order]
    # scaling denominator
    du0 = u0.max()-u0.min()
    
    # point values
    vv = np.array([u0+2*u1, u0-u1+u2, u0-u1-u2])
    ww = []
    for i in range(mesh.ne):
        bNbr = listNbw[i]
        w0 = []
        for j in range(len(bNbr)):
            w0.append(vv[:,i].dot(bNbr[j]))
        ww.append(w0)
        
    # revert extended cell averages to current cell
    uu = []
    for i in range(mesh.ne):
        nbr = listNbr[i]
        nbj = listNbj[i]
        uu0 = []
        bcflag = 0
        for j in range(len(nbr)):
            # HACK!!! for bdry cell avg
            if nbr[j] !=-1:
                uu0.append(ww[nbr[j]][nbj[j]])
            else: # bdry edge::: HACK append own cell avg
                uu0.append(u0[i])
                bcflag = 1
        uu.append(uu0)
    uu = np.array(uu).T
    diff_u = np.abs(uu[0]-u0)+np.abs(uu[1]-u0)+np.abs(uu[2]-u0)
    
    if flag=="du":
        ind = diff_u/(du0+1e-8*u0.max()) # regularize denominator
    else:
        print("Please provide a good scaling for Indicator2D")
        stop
    return ind


######### PART 3: LIMITERS 

#### vetorized minmod function
def minmod(a1, a2, a3, M=0):
    sa0 = np.abs(a1) < M # these cells does not change
    
    sa1 = np.sign(a1)
    sa2 = np.sign(a2)
    sa3 = np.sign(a3)
    s4 = np.logical_and(sa1==sa2, sa1==sa3)
    vals = np.array([abs(a1),abs(a2),abs(a3)])
    mina = vals.min(axis=0)
    
    tmp = s4*sa1*mina
    tmp[sa0] = a1[sa0]
    return tmp

# minmod on triangles 
# ref [CockburnShu RKDG JSC review article]
def minmod2D(vv,ww, eps0=1e-12):
    sv = np.sign(vv)
    sw = np.sign(ww)
    _, nL = sw.shape

    s0 = sv==sw # the sign
    vals = np.array([abs(vv.reshape(1, 3*nL)[0]), 
                    abs(ww.reshape(1, 3*nL)[0])])
    mina = vals.min(axis=0).reshape(3, nL)

    # minmod values
    zz = s0*sv*mina
    sum_zz =zz.sum(axis=0)
    
    # determin unmodified cells
    tmp = zz==vv
    # cells to be modified ?!?!
    maskX = ~(tmp[0,:]*tmp[1,:]*tmp[2,:])    

    # the zeros
    mask = abs(sum_zz) < eps0
    # the pos/neg
    sz = zz[:,~mask]>0
    zp = sz*zz[:,~mask]
    zm = zp-zz[:,~mask]
    sum_zp =zp.sum(axis=0)
    sum_zm =zm.sum(axis=0)

    # the thetas: TODO: avoid division by zero add regularizer 1e-40?!?!
    maskP = abs(sum_zp) > eps0 
    theta = sum_zm/(sum_zp+1e-40)
    id0 = theta > 1   
    tp = theta
    tp[id0] = 1
    
    tm = sum_zp/(sum_zm+1e-40)
    tm[~id0] = 1
    
    # recover
    for j in range(3):
        zz[j, ~mask] = tp*zp[j,:]-tm*zm[j,:]
    return maskX, zz


# SW system limter 1D
def tvbLimitM1D(rhoh, mh, bh, k0, nelems, listNbs, tol=0.2, flag="comp", g = 9.812, M = 0, 
             epsRho=1e-6, limitB=True):
    if k0 == 0:
        return 0, 0
    # only limit wet cells
    mask2 = rhoh.vec[:nelems].FV().NumPy() >= epsRho
    
    # (den+b) -indicator
    rhoh.vec.data += bh.vec
    ind0 = computeIndicator1D(rhoh, k0, listNbs)
    mask1 = ind0 > tol
    
    mask = np.logical_and(mask1, mask2)
    
    if limitB != True: # limit (rho,m)
        rhoh.vec.data -= bh.vec
             
    nlimit0 = np.count_nonzero(mask)
    ### component-wise limiting: rhoh and mh 
    if flag=="comp":
        for gf in [rhoh, mh]:
            ###### den-minmod
            u0 = gf.vec[0:nelems].FV().NumPy()
            # higher order terms
            view = gf.vec[nelems:].FV().NumPy().reshape(nelems, k0)    

            ### limit masked cell dofs (TVD)
            u1 = view[mask,0] # P1 part of the soln (slope)
            uL0 = -u0[listNbs[0]][mask] # (-) left nbrs
            uR0 = u0[listNbs[1]][mask] # right nbrs
            uL0 += u0[mask] # left slope
            uR0 -= u0[mask] # right slope
            # limit
            v1 = minmod(u1, uL0, uR0)
            maskX = (v1 != u1)
            ids = np.arange(0, nelems, 1)
            ids0 = ids[mask]
            ids1 = ids0[maskX]
            nlimit1 = np.count_nonzero(maskX)
            
            view[ids1,0] = v1[maskX]
            view[ids1,1:] = 0 # set higher order (>P2) parts to zero
    else:
        # characteristic decomp on masked cells
        # cell averages
        rho0 = rhoh.vec[0:nelems].FV().NumPy()
        rho1 = rhoh.vec[nelems:].FV().NumPy().reshape(nelems, k0)    
        
        m0 = mh.vec[0:nelems].FV().NumPy()
        m1 = mh.vec[nelems:].FV().NumPy().reshape(nelems, k0)    
        
        # left/right slopes averages
        rhoL0 = rho0[mask]-rho0[listNbs[0]][mask] # (-) left nbrs
        rhoR0 = rho0[listNbs[1]][mask]-rho0[mask] # right nbrs

        # left/right cell averages
        mL0 = m0[mask]-m0[listNbs[0]][mask] # (-) left nbrs
        mR0 = m0[listNbs[1]][mask] - m0[mask] # right nbrs

        # velocity avg on masked cells
        u0 = m0[mask]/rho0[mask]
        lam1 = u0 - (g*rho0[mask])**0.5
        lam2 = u0 + (g*rho0[mask])**0.5
                
        #    R = 1/sqrt(2 g)[1 1; lam1, lam2]
        # invR = 1/sqrt(2 rho)[lam2, -1; -lam1, 1]
        vL0 = (lam2*rhoL0-mL0)/(2*rho0[mask])**0.5
        v0 = (lam2*rho1[mask,0]-m1[mask,0])/(2*rho0[mask])**0.5        
        vR0 = (lam2*rhoR0-mR0)/(2*rho0[mask])**0.5
        # limit slope
        w0 = minmod(v0, vL0, vR0, M)

        vL1 = -(lam1*rhoL0-mL0)/(2*rho0[mask])**0.5
        v1 = -(lam1*rho1[mask,0]-m1[mask,0])/(2*rho0[mask])**0.5        
        vR1 = -(lam1*rhoR0-mR0)/(2*rho0[mask])**0.5
        # limit slope
        w1 = minmod(v1, vL1, vR1, M)
        
        # update slope R*v
        rhoX = (w0+w1)/(2*g)**0.5
        mX = (lam1*w0+lam2*w1)/(2*g)**0.5
        
        # modified slopes
        maskX = np.logical_or((w0 != v0), (w1 != v1))
        ids = np.arange(0, nelems, 1)
        ids0 = ids[mask]
        ids1 = ids0[maskX]
        nlimit1 = np.count_nonzero(maskX)

        # only limit cells with different slope
        rho1[ids1,0] = rhoX[maskX]
        m1[ids1,0] = mX[maskX]
        rho1[ids1,1:]=0
        m1[ids1,1:]=0                
    if limitB == True: # limit (rho,m)
        rhoh.vec.data -= bh.vec
    return mask, nlimit0

def tvbLimitM2D(rhoh, bh, mh1, mh2, order, mesh, listNbr, listNbj, listNbw, alphaList, listNrms, nu=1.5, 
                eps0=1e-12, tol=0.05, flag="char", g=9.812, limitB=True, epsRho=1e-6):
    if order ==0:
        return 0, 0
    ### Step 1: compute indicator based on rho+bh
    # only limit wet cells
    mask2 = rhoh.vec[:mesh.ne].FV().NumPy() >= epsRho
        
    rhoh.vec.data += bh.vec
    indicator = computeIndicator2D(rhoh, order, mesh, listNbr, listNbj, listNbw)
    
    # find the cells to be limited
    mask1 = indicator > tol
    maskL = np.logical_and(mask1, mask2)
    
    if limitB != True: # limit (rho,m)
        rhoh.vec.data -= bh.vec
        
    if sum(maskL)==False: # no cells to be limited
        if limitB == True: # recovery rhoh
            rhoh.vec.data -= bh.vec
        return 0, 0    

    # the masked cell numbers
    ind0 = np.array([i for i in range(mesh.ne)])
    indM = ind0[maskL]
    
    ### Step 2: prepare data cell averages and linear parts
    hdof = int((order+3)*order/2)
    
    ZZZ = [rhoh, mh1, mh2]
    U0, VV, WW = [], [], [] # data to be limited 
    for val in ZZZ:
        ### 2.1 water height
        V0 = val.vec[:mesh.ne].FV().NumPy()
        # cell averages
        Vho = val.vec[mesh.ne:].FV().NumPy().reshape(mesh.ne, hdof)
        
        u0 = V0[maskL]
        u1 = Vho[maskL,0]
        u2 = Vho[maskL,order]
        ## NOTE: THIS IS SPECIFIC TO L2 basis implemented in NGSOLVE
        # facs = [[1 0.5, -0.5], [1,-1,0],[1,0.5, 0.5]]
        # the tildes on maskL cells (original slopes)
        vv = np.array([0.5*u1-0.5*u2, -u1, 0.5*u1+0.5*u2])

        # limited slopes
        ww = []
        for j0, i in enumerate(indM):
            b0 = u0[j0]
            list0 = listNbr[i]
            bNbr = V0[list0]-b0
            alpha = alphaList[i]
                
            w0 = []
            for j in range(3):
                i0 = alpha[j][2]
                if i0 == -1 or list0[j] == -1: # HACK bdry edge/ no slope
                    w0.append(0)
                else:
                    w0.append(bNbr[i0]*alpha[j][0]+bNbr[(i0+1)%3]*alpha[j][1])
            ww.append(w0)
        ww = nu*np.array(ww).T
                
        if flag =="comp": # component-wise limiting (THIS IS FINE)
            # apply minmod limiter
            maskX, zz = minmod2D(vv, ww, eps0)
            # update h.o. parts
            Vho[indM[maskX],:] = 0   
            Vho[indM[maskX],0] = -zz[1,maskX]
            Vho[indM[maskX],order] = -zz[0,maskX]+zz[2,maskX]
        else:
            #### save data for characteristic decomp
            U0.append(u0)
            VV.append(vv)
            WW.append(ww)

    if flag=="comp": # finished limiting
        if limitB ==True: # recover rho
            rhoh.vec.data -= bh.vec
        return maskX, np.count_nonzero(maskX)       
    
    # apply characteristic-wise limiting: averaging three directions
    nL = len(indM) # total cells to be limited
    
    ## LOOP OVER THREE EDGES
    for j0 in range(3):
        # looping: NOT EFFICIENT
        W0, W1, W2 = [],[],[]
        V0, V1, V2 = [],[],[]    
        for i in range(nL):
            h = U0[0][i]
            u = U0[1][i]/h
            v = U0[2][i]/h
            c = (g*h)**0.5
            n = listNrms[indM[i]][j0] # normal direction
            nx, ny = n[0], n[1]
            un = u*nx+v*ny

            fac = 1/(2*h)**0.5
            w0 = fac*(       (c-un)*WW[0][:,i]  + nx*WW[1][:,i]  +ny*WW[2][:,i])
            w1 = fac*(2*(u*ny-v*nx)*WW[0][:,i]- 2*ny*WW[1][:,i]+2*nx*WW[2][:,i])
            w2 = fac*(       (c+un)*WW[0][:,i]  - nx*WW[1][:,i]  -ny*WW[2][:,i])

            v0 = fac*(       (c-un)*VV[0][:,i]  + nx*VV[1][:,i]  +ny*VV[2][:,i])
            v1 = fac*(2*(u*ny-v*nx)*VV[0][:,i]- 2*ny*VV[1][:,i]+2*nx*VV[2][:,i])
            v2 = fac*(       (c+un)*VV[0][:,i]  - nx*VV[1][:,i]  -ny*VV[2][:,i])

            W0.append(w0)
            W1.append(w1)
            W2.append(w2)

            V0.append(v0)
            V1.append(v1)
            V2.append(v2)

        # characteristic-wise limit
        mask0, Z0 = minmod2D(np.array(V0).T, np.array(W0).T)
        mask1, Z1 = minmod2D(np.array(V1).T, np.array(W1).T)
        mask2, Z2 = minmod2D(np.array(V2).T, np.array(W2).T)

        # convert back * R
        X0, X1, X2 = [],[],[]
        for i in range(nL):
            h = U0[0][i]
            u = U0[1][i]/h
            v = U0[2][i]/h
            c = (g*h)**0.5        
            n = listNrms[indM[i]][j0] # normal direction
            nx, ny = n[0], n[1]
            un = u*nx+v*ny

            fac = 1/(2*g)**0.5
            w0 = fac*(         Z0[:,i]                   +      Z2[:,i])
            w1 = fac*((u+c*nx)*Z0[:,i] - c*ny*Z1[:,i]+ (u-c*nx)*Z2[:,i])
            w2 = fac*((v+c*ny)*Z0[:,i] + c*nx*Z1[:,i]+ (v-c*ny)*Z2[:,i])

            X0.append(w0)
            X1.append(w1)
            X2.append(w2)
        if j0 == 0:
            XX = [1/3*np.array(X0).T, 1/3*np.array(X1).T, 1/3*np.array(X2).T]
        else:
            XX[0] += 1/3*np.array(X0).T
            XX[1] += 1/3*np.array(X1).T
            XX[2] += 1/3*np.array(X2).T
    
    # limit quantities
    for i, val in enumerate(ZZZ):
        # cell averages
        Vho = val.vec[mesh.ne:].FV().NumPy().reshape(mesh.ne, hdof)
        zz = XX[i]
        Vho[indM,:] = 0   
        Vho[indM,0] = -zz[1,:]
        Vho[indM,order] = -zz[0,:]+zz[2,:]
    if limitB == True: # recovery rhoh
        rhoh.vec.data -= bh.vec
    return maskL, np.count_nonzero(maskL)


# scalar TVB limiter on trig
def tvbLimit2D(gfu, order, mesh, listNbr, listNbj, listNbw, alphaList, nu=1.5, eps0=1e-14, tol=0.1):
    ### Step 1: compute indicator
    indicator = computeIndicator2D(gfu, order, mesh, listNbr, listNbj, listNbw)
    # find the cells to be limited
    maskL = indicator > tol
    if sum(maskL)==False: # no cells to be limited
        return 0
    
    ### Step 2: prepare data cell averages and linear parts
    U0 = gfu.vec[:mesh.ne].FV().NumPy()
    hdof = int((order+1)*(order+2)/2-1)
    # cell averages
    uho = gfu.vec[mesh.ne:].FV().NumPy().reshape(mesh.ne, hdof)
    
    # P0/P1 parts on maskL cells
    u0 = U0[maskL]
    u1 = uho[maskL,0]
    u2 = uho[maskL,order]
    
#     facs = [[1 0.5, -0.5], [1,-1,0],[1,0.5, 0.5]]
    # the tildes on maskL cells
    vv = np.array([0.5*u1-0.5*u2, -u1, 0.5*u1+0.5*u2])
        
    # the masked cell numbers
    ind0 = np.array([i for i in range(mesh.ne)])
    indM = ind0[maskL]
    
    ww = []
    for j0, i in enumerate(indM):
        b0 = u0[j0]
        bNbr = U0[listNbr[i]]-b0
        alpha = alphaList[i]
        w0 = []
        for j in range(3):
            i0 = alpha[j][2]
            if i0 == -1 or list0[j] == -1: # HACK bdry edge/ no slope
                w0.append(0)
            else:
                w0.append(bNbr[i0]*alpha[j][0]+bNbr[(i0+1)%3]*alpha[j][1])
        ww.append(w0)
    ww = nu*np.array(ww).T
    
    # apply minmod limiter
    maskX, zz = minmod2D(vv, ww, eps0)

    # update h.o. parts
    uho[indM[maskX],:] = 0   
    uho[indM[maskX],0] = -zz[1,maskX]
    uho[indM[maskX],order] = -zz[0,maskX]+zz[2,maskX]
    return np.count_nonzero(maskX)


# positivity preserving limiter for density (water height)
def pplimit(gf, order, mips, npts, nelems, mw, wt=1, dim=1, eps0 = 1e-12):
    if order==0:
        return 0
    u0 = gf.vec[0:nelems].FV().NumPy()
    vals = gf(mips).reshape(nelems, npts) # LEFT/RIGHT values
    if order > 1:
        #### NOTE: vals are located on bdry only
        if dim ==1: # 1D case
            umid = (u0-mw*(vals[:,0]+vals[:,1]))/(1-2*mw)
        elif dim==2: # 2D case (include mid pt) ?!?!
            umid = u0/(1-2*mw)
            for i, w in enumerate(wt):
                umid -= 2./3.*mw/(1-2*mw)*w*(vals[:,3*i]+vals[:,3*i+1]+vals[:,3*i+2])
        vals = np.column_stack((vals, umid))
    
    umin = vals.min(axis=1)
    # TODO make it more efficient (ONLY MODIFY TROUBLED CELLS)
    # Question: why diff<0?!
    diff = np.abs(u0-umin)
    mask0 =  (diff > 1e-14) # THIS ARE CELLS TO BE CHECKED
    
    # avoid division by zero
    tt = np.array([1 for i in range(nelems)])
    tt[mask0] = (u0[mask0]-eps0) / diff[mask0]

    mask = tt<1
    
    if sum(mask)==0:
        return mask

    # dofs excluding cell averages
    if dim ==1:
        ho = order
    elif dim==2:
        ho = int((order+3)*order/2)
    view = gf.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    view[mask] *= tt[mask, np.newaxis]
    
    return mask

# Remove high-order dofs on dry cells for SW
# dry cell is determined by rho_avg < epsRho
def drylimit1D(rhoh, mh, order, nelems, epsRho = 1e-6):
    # convert to P0 on dry cells
    rho0 = rhoh.vec[0:nelems].FV().NumPy()
    mask = rho0 < epsRho
    # Go to P0
    rho1 = rhoh.vec[nelems:].FV().NumPy().reshape(nelems, order)
    rho1[mask] = 0
    m1 = mh.vec[nelems:].FV().NumPy().reshape(nelems, order)
    m1[mask] = 0

def drylimit2D(rhoh, mh1, mh2, order, nelems, epsRho = 1e-6):
    # convert to P0 on dry cells
    rho0 = rhoh.vec[0:nelems].FV().NumPy()
    mask = rho0 < epsRho
    ho = int(order*(order+3)/2)
    # Go to P0
    rho1 = rhoh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    rho1[mask] = 0
    m1 = mh1.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    m1[mask] = 0
    m2 = mh2.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    m2[mask] = 0

    
# velocity limiting: 1D case
def limitU1D(uh, nelems, order, mat, maxV=9.0):
    u0 = uh.vec[:nelems].FV().NumPy()
    ## TODO: user adjusted uMax, uMin
    uMax, uMin = maxV, -maxV

    if order >0:
        u1 = uh.vec[nelems:].FV().NumPy().reshape(nelems, order)
        uA = u1.dot(mat)
        uA += u0[:,np.newaxis]

        uP = uA.max(axis=1)
        uM = uA.min(axis=1)
        # These are troubled cells
        mask = np.logical_or(uP > uMax, uM < uMin)
        #(1) set-zero ho parts
        u1[mask] = 0
    else:
        mask = np.logical_or(u0 > uMax, u0 < uMin)

    #(2) hack cell averages
    # set-zero vel average
    u0[mask] = 0

    ind0 = np.array([i for i in range(nelems)])
    indM = ind0[mask]

    # vel recovery by borrowing neighbor values
    while len(indM)>0:
        badM = []
        for j0, i in enumerate(indM):
            b0 = 0
            ll = 0
            iL = i-1
            iR = i+1
            if iL != 0 and (iL not in indM): # good cell avg
                ll += 1
                b0 += u0[iL]
            if iR != nelems-1 and (iR not in indM): # good cell avg
                ll += 1
                b0 += u0[iL]
            if ll ==0:
                badM.append(i)
            else:
                u0[i] = b0/ll # replace vel by average 
        indM = badM
    return np.count_nonzero(mask)     
    
    
# velocity limiting 2D case
def limitU2D(uh1, uh2, nelems, order, mat, listNbr, maxV=9.0):
    for uu in [uh1, uh2]:
        u0 = uu.vec[:nelems].FV().NumPy()
        ## TODO: user adjusted uMax, uMin
        uMax, uMin = maxV, -maxV
                
        if order >0:
            u1 = uu.vec[nelems:].FV().NumPy().reshape(nelems, int((order**2+3*order)/2))
            uA = u1.dot(mat)
            uA += u0[:,np.newaxis]

            uP = uA.max(axis=1)
            uM = uA.min(axis=1)
            # These are troubled cells
            mask = np.logical_or(uP > uMax, uM < uMin)
            #(1) set-zero ho parts
            u1[mask] = 0
        else:
            mask = np.logical_or(u0 > uMax, u0 < uMin)
                
        #(2) hack cell averages
        # set-zero vel average
        u0[mask] = 0
        
        ind0 = np.array([i for i in range(nelems)])
        indM = ind0[mask]
        
        # vel recovery by borrowing neighbor values
        while len(indM)>0:
            badM = []
            for j0, i in enumerate(indM):
                list0 = listNbr[i]
                b0 = 0
                ll = 0
                for j in range(3):
                    if list0[j] != -1 and (list0[j] not in indM): # good cell avg
                        ll += 1
                        b0 += u0[list0[j]]
                if ll ==0:
                    badM.append(i)
                else:
                    u0[i] = b0/ll # replace vel by average 
            indM = badM
    return np.count_nonzero(mask)             
    
    
    
    
    
    
    
    
    
# positivity preserving limiter for compressible Euler in 1D
def pplimitEuler1D(gf, order, mips, npts, nelems, mw, wt=1, dim=1, eps0 = 1e-14, gamma1=0.4):
    if order==0:
        return 0
    # dofs excluding cell averages
    ho = order # number of ho dofs
        
    rhoh, mh, Eh = gf.components
    # cell averages
    rho0 = rhoh.vec[0:nelems].FV().NumPy()
    m0 = mh.vec[0:nelems].FV().NumPy()
    E0 = Eh.vec[0:nelems].FV().NumPy()
    p0 = gamma1*(E0-0.5*m0**2/rho0)
    
    # the ho DOFS
    rhoview = rhoh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    mview = mh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    Eview = Eh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
    
    # nodal values
    rho1 = rhoh(mips).reshape(nelems, npts) # LEFT/RIGHT values
    m1 = mh(mips).reshape(nelems, npts) # LEFT/RIGHT values
    E1 = Eh(mips).reshape(nelems, npts) # LEFT/RIGHT values
    if order > 1:
        #### NOTE: vals are located on bdry only
        rhomid = (rho0-mw*(rho1[:,0]+rho1[:,1]))/(1-2*mw)
        mmid = (m0-mw*(m1[:,0]+m1[:,1]))/(1-2*mw)
        Emid = (E0-mw*(E1[:,0]+E1[:,1]))/(1-2*mw)
        
        rho1 = np.column_stack((rho1, rhomid))
        m1 = np.column_stack((m1, mmid))
        E1 = np.column_stack((E1, Emid))
    
    ### Step 1: limit density
    rhomin = rho1.min(axis=1)
    # TODO make it more efficient (ONLY MODIFY TROUBLED CELLS)
    diff = np.abs(rho0-rhomin)
    mask0 = (diff != 0)
    
    # avoid division by zero
    tt = np.array([1 for i in range(nelems)])
    tt[mask0] = (rho0[mask0]-eps0) / diff[mask0]

    mask = tt<1   
    # LIMIT 1: update den ho dofs
    rhoview[mask] *= tt[mask, np.newaxis]
    
    # update den @ quad points
    scale = (1-tt[mask])*rho0[mask]
    rho1[mask,:] += scale[:, np.newaxis]
    
    # step 2: limit pressure
    p1 = gamma1*(E1-0.5*m1**2/rho1)
    pmin = p1.min(axis=1)
    maskP = pmin < 0
    theta = p0[maskP]/(p0[maskP]-pmin[maskP])
        
    rhoview[maskP, :] *= theta[:, np.newaxis]
    mview[maskP, :] *= theta[:, np.newaxis]
    Eview[maskP, :] *= theta[:, np.newaxis]
    
    return mask


# tvb limiter for compressible Euler in 1D
def tvblimitEuler1D(gfu, order, nelems, listNbs, tol=0.02, flag="comp", gamma=1.4, gamma1=0.4):
    if order==0:
        return np.array([False]), 0, 0
    # dofs excluding cell averages
    ho = order # number of ho dofs    
    rhoh, mh, Eh = gfu.components
    
    # den-based indicator
    ind0 = indicator(rhoh, order, listNbs)
    ind1 = indicator(Eh, order, listNbs)
    maskA = ind0 > tol
    maskB = ind1 > tol
    mask = np.logical_or(maskA, maskB)
    
    nlimit0 = np.count_nonzero(mask)
    
    if flag=="comp":
        for gf in [rhoh, mh, Eh]:
            ###### den-minmod
            u0 = gf.vec[0:nelems].FV().NumPy()
            # higher order terms
            view = gf.vec[nelems:].FV().NumPy().reshape(nelems, ho)    

            ### limit masked cell dofs (TVD)
            u1 = view[mask,0] # P1 part of the soln (slope)
            uL0 = u0[mask] - u0[listNbs[0]][mask] # (-) left nbrs
            uR0 = -u0[mask] + u0[listNbs[1]][mask] # right nbrs
            
            # limit
            v1 = minmod(u1, uL0, uR0)
            maskX = (v1 != u1)
            ids = np.arange(0, nelems, 1)
            ids0 = ids[mask]
            ids1 = ids0[maskX]
            nlimit1 = np.count_nonzero(maskX)
            
            view[ids1,0] = v1[maskX]
            view[ids1,1:] = 0 # set higher order (>P2) parts to zero
    else:
        # characteristic decomp on masked cells
        # cell averages
        rho0 = rhoh.vec[0:nelems].FV().NumPy()
        m0 = mh.vec[0:nelems].FV().NumPy()
        E0 = Eh.vec[0:nelems].FV().NumPy()
        
        rho1 = rhoh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
        m1 = mh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
        E1 = Eh.vec[nelems:].FV().NumPy().reshape(nelems, ho)
        
        # left/right slopes averages
        rhoL0 = rho0[mask]-rho0[listNbs[0]][mask] # (-) left nbrs
        rhoR0 = rho0[listNbs[1]][mask]-rho0[mask] # right nbrs

        # left/right cell averages
        mL0 = m0[mask]-m0[listNbs[0]][mask] # (-) left nbrs
        mR0 = m0[listNbs[1]][mask] - m0[mask] # right nbrs

        # left/right cell averages
        EL0 = E0[mask]-E0[listNbs[0]][mask] # (-) left nbrs
        ER0 = E0[listNbs[1]][mask] - E0[mask] # right nbrs
        
        # velocity/pres 
        u0 = m0[mask]/rho0[mask]
        p0 = gamma1*(E0[mask]-0.5*m0[mask]*u0)
        c0 = (gamma*p0/rho0[mask])**0.5
        H0 = (E0[mask]+p0)/rho0[mask]
        B1 = gamma1/gamma*rho0[mask]/p0
        B2 = 0.5*B1*u0**2
        
        # characteristic decomposition
        # R = [1 1 1; u-c, u, u+c; H-cu, 0.5 u*u, H+cu]
        #
        # invR = [0.5(B2+u/c) -0.5(B1 u+1/c) 0.5B1]
        #        [1-B2,                B1 u, -B1  ]
        #        [0.5(B2-u/c),-0.5(B1 u-1/c), 0.5B1]

        
        #        [0.5(B2+u/c) -0.5(B1 u+1/c) 0.5B1]
        vL0 = 0.5*(B2+u0/c0)*rhoL0      -0.5*(B1*u0+1/c0)*mL0      + 0.5*B1*EL0
        v0  = 0.5*(B2+u0/c0)*rho1[mask,0] -0.5*(B1*u0+1/c0)*m1[mask,0] + 0.5*B1*E1[mask,0]
        vR0 = 0.5*(B2+u0/c0)*rhoR0      -0.5*(B1*u0+1/c0)*mR0      + 0.5*B1*ER0
        # limit slope
        w0 = minmod(v0, vL0, vR0)

        #        [1-B2,                B1 u, -B1  ]
        vL1 = (1-B2)*rhoL0      +B1*u0*mL0      -B1*EL0
        v1  = (1-B2)*rho1[mask,0] +B1*u0*m1[mask,0] -B1*E1[mask,0]
        vR1 = (1-B2)*rhoR0      +B1*u0*mR0      -B1*ER0
        # limit slope
        w1 = minmod(v1, vL1, vR1)

        #        [0.5(B2-u/c),-0.5(B1 u-1/c), 0.5B1]
        vL2 = 0.5*(B2-u0/c0)*rhoL0      -0.5*(B1*u0-1/c0)*mL0      + 0.5*B1*EL0
        v2  = 0.5*(B2-u0/c0)*rho1[mask,0] -0.5*(B1*u0-1/c0)*m1[mask,0] + 0.5*B1*E1[mask,0]
        vR2 = 0.5*(B2-u0/c0)*rhoR0      -0.5*(B1*u0-1/c0)*mR0      + 0.5*B1*ER0
        # limit slope
        w2 = minmod(v2, vL2, vR2)
        
        # update slope R*v
        # R = [1 1 1; u-c, u, u+c; H-cu, 0.5 u*u, H+cu]
        rhoX = w0+w1+w2
        mX = (u0-c0)*w0+u0*w1 + (u0+c0)*w2
        EX = (H0-c0*u0)*w0+0.5*u0*u0*w1 + (H0+c0*u0)*w2
        
        # modified slopes
        maskX = np.logical_or(np.logical_or((w0 != v0), (w1 != v1)), w2 !=v2)
        
        ids = np.arange(0, nelems, 1)
        ids0 = ids[mask]
        ids1 = ids0[maskX]
        nlimit1 = np.count_nonzero(maskX)

        # only limit cells with different slope
        rho1[ids1,0] = rhoX[maskX]
        m1[ids1,0] = mX[maskX]
        E1[ids1,0] = EX[maskX]
        rho1[ids1,1:]=0
        m1[ids1,1:]=0                
        E1[ids1,1:]=0
    return mask, nlimit0, nlimit1