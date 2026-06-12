import numpy as np
from numpy import pi, sinh, cosh, tanh, cos, sqrt, arctan, sin
from mpmath import sech, besselk
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy import optimize

# This is taken from the open source gw_hyp library: https://github.com/subhajitphy/GW_hyp/tree/main
G=6.67408*1e-11
Ms=1.898*1e+30
c=3e8

tsun=4.691631051851851e-06
dsun=1407.48931
yr=365.25*24*3600

pc=3.086e+16

#Define in which Post Newtonian Order we are interested in, for eg. order=n refers to nPN order.
def PNx(pn0,pn1,pn2,pn3,initial,x,order):
    s=0
    pn=[pn0,pn1,pn2,pn3]
    for i in range(order+1):
        s+=pn[i]*x**(initial+i)
    return s


def rx(eta,et,u,x,order):
    
    a0=(-1+et*cosh(u))

    a1= 1/6*(-18+2*eta+et*(-6+7*eta)*cosh(u))
    a2=((-216+534*eta+8*eta**2-2*et**2*(4*eta**2+15*eta+36)+et*(et**2-1)*(35*eta**2-
    231*eta+72)*cosh(u))/(72*et**2-72))

    a3=(1/181440/(et**2-1)**2*(-4233600+12143736*eta-348705*pi**2*eta-761040*eta
    **2+4480*eta**3+280*et**4*(16*eta**3+90*eta**2-81*eta+432)-et**2*(3144960+81*(1435*pi
    **2-134336)*eta+3437280*eta**2+8960*eta**3)+140*et*(et**2-1)**2*(49*eta**3-3933*eta**2
    +7047*eta-864)*cosh(u)))


    return PNx(a0,a1,a2,a3,-1,x,order)


def rtx(eta,et,u,x,order):
    a0=et*sinh(u)/(-1+et*cosh(u))
    a1=sinh(u)*et*(-6+7*eta)/(-6+6*et*cosh(u))
    a2=(35/72*et*(et**3*(eta**2-33/5*eta+72/35)*cosh(u)**3-3*et**2*(eta**2-33/5*eta+72/35)*
    cosh(u)**2+96/35*et*(eta**2-57/16*eta-27/8)*cosh(u)+(9/35*eta**2-27/7*eta)*et**2-
    eta**2+3/7*eta+468/35)*sinh(u)/(-1+et*cosh(u))**4)
    a3=(41/64*et*(196/3321*(et-1)*(et+1)*(eta**3-3933/49*eta**2+7047/49*eta-864/49)*et**5*
    cosh(u)**5-980/3321*(et-1)*(et+1)*(eta**3-3933/49*eta**2+7047/49*eta-864/49)*et**4*
    cosh(u)**4-2*((-1196/3321*eta**3+10720/369*eta**2-2436/41*eta+1360/123)*et**2+560/
    123+1196/3321*eta**3-7408/369*eta**2+(pi**2+42796/4305)*eta)*et**3*cosh(u)**3+et**2*(
    (80/123*eta**3-232/41*eta**2+192/41*eta)*et**4+(-2080/123-8224/3321*eta**3+24680/
    369*eta**2+(pi**2-95696/1435)*eta)*et**2+13600/123+6064/3321*eta**3-2720/369*eta**2+
    (-336992/1435+5*pi**2)*eta)*cosh(u)**2-2*((-58/123*eta**3-338/41*eta**2+1462/41*eta
    )*et**4+(-5200/123+1346/3321*eta**3+7538/369*eta**2+(pi**2-20166/1435)*eta)*et**2+
    10960/123+220/3321*eta**3+5440/369*eta**2+(-243988/1435+2*pi**2)*eta)*et*cosh(u)+(
    -92/41*eta-52/41*eta**3+292/41*eta**2)*et**6+(3008/41*eta+272/123*eta**3-1320/41*
    eta**2)*et**4+(-6112/123-3328/3321*eta**3+12728/369*eta**2+(pi**2-1856/35)*eta)*et**2
    +9952/123+196/3321*eta**3+3148/369*eta**2+(-14396/123+pi**2)*eta)*sinh(u)/(et**2-1)
    /(-1+et*cosh(u))**6)
    return PNx(a0,a1,a2,a3,1/2,x,order)


def phitx(eta,et,u,x,order):
    a0=(et**2-1)**(1/2)/(-1+et*cosh(u))**2
    a1=((et*(-1+eta)*cosh(u)-3+(-eta+4)*et**2)/(et**2-1)**(1/2)/(-1+et*cosh(u))**3)
    a2=(1/12*(-14*et**3*((eta**2+5*eta-3/7)*et**2-4/7*eta**2+1/7*eta-18/7)*cosh(u)**3+17*et**
    2*((48/17+eta**2-eta)*et**4+(-66/17-4/17*eta**2+8*eta)*et**2-108/17+5/17*eta**2+97/
    17*eta)*cosh(u)**2-et*(et**4*(eta**2-97*eta-12)+(16*eta**2+188*eta+102)*et**2+eta**2+
    125*eta-216)*cosh(u)+(-12*eta**2-18*eta)*et**6+(20*eta**2-26*eta-60)*et**4+(-2*eta**
    2+68*eta+162)*et**2+48*eta-144)/(et**2-1)**(3/2)/(-1+et*cosh(u))**5)
    a3=(1/6720*(12915*((64/123*eta**3+2912/369*eta**2-104/9*eta+64/41)*et**4+(736/41-160/
    369*eta**3+2432/369*eta**2+(pi**2-216592/4305)*eta)*et**2+128/41+64/369*eta**3+128/
    369*eta**2+(1/3*pi**2-1112/123)*eta)*et**5*cosh(u)**5+8610*et**4*((-116/123*eta**3+92
    /123*eta**2+44/41*eta)*et**6+(736/41+116/123*eta**3-2564/123*eta**2+(pi**2-4448/105)
    *eta)*et**4+(-5200/41-132/41*eta**3-12332/123*eta**2+(370220/861-11/2*pi**2)*eta)*
    et**2-2496/41+52/41*eta**3+1124/123*eta**2+(616088/4305-11/2*pi**2)*eta)*cosh(u)**4-
    34440*et**3*((-1/123*eta**3+245/123*eta**2-694/123*eta-144/41)*et**6+(1096/41-55/
    123*eta**3+53/123*eta**2+(pi**2-204698/4305)*eta)*et**4+(-2248/41-119/123*eta**3-
    7705/123*eta**2+(-7/4*pi**2+159434/861)*eta)*et**2-2184/41+55/123*eta**3+189/41*eta
    **2+(-17/4*pi**2+576538/4305)*eta)*cosh(u)**3-8610*((-32/123*eta**3+48/41*eta**2-40/
    41*eta)*et**8+(1728/41-28/123*eta**3-908/41*eta**2+(pi**2-172708/4305)*eta)*et**6+(-
    7776/41+180/41*eta**3-1316/123*eta**2+(-9*pi**2+2116892/4305)*eta)*et**4+(6816/41-
    12/41*eta**3+30772/123*eta**2+(6*pi**2-1072548/1435)*eta)*et**2+13152/41+12/41*eta**
    3+484/123*eta**2+(22*pi**2-660244/861)*eta)*et**2*cosh(u)**2+17220*et*(-8/123*eta*(
    eta**2+75*eta+379/2)*et**8+(864/41+34/41*eta**3+1538/123*eta**2+(pi**2-132248/4305)*
    eta)*et**6+(-2912/41-118/123*eta**3-3974/123*eta**2+(-5*pi**2+229366/861)*eta)*et**4
    +(968/41+146/123*eta**3+9142/123*eta**2+(11/4*pi**2-376388/1435)*eta)*et**2+4560/41
    -2/123*eta**3+734/123*eta**2+(25/4*pi**2-977078/4305)*eta)*cosh(u)+(5040*eta**3+
    20160*eta**2-7560*eta)*et**10+(-26320*eta**3-6720*eta**2+241640*eta)*et**8+(-120960+
    42000*eta**3-142240*eta**2+(-8610*pi**2-95584)*eta)*et**6+(336000-26320*eta**3+
    355040*eta**2+(34440*pi**2-1401544)*eta)*et**4+(-3360+2240*eta**3-404320*eta**2+(-
    21525*pi**2+1368504)*eta)*et**2-21525*pi**2*eta-13440*eta**2+810320*eta-504000)/(et
    **2-1)**(5/2)/(-1+et*cosh(u))**7)
    return PNx(a0,a1,a2,a3,3/2,x,order)


def vH(eta,et,u,x,order):
    ephi=ephiet(eta,et,x,order)*et
    return (2*arctan(((ephi+1)/(ephi-1))**(1/2)*tanh(1/2*u)))


def ephiet(eta,et,x,order):
    a0= 1
    a1= (-4+eta)
    a2=((4*eta**2+260*eta-2016+et**2*(41*eta**2-659*eta+1152))/(96*et**2-96))
    a3=(1/26880/(et**2-1)**2*(-4139520-20*(861*pi**2-178748)*eta+155680*eta**2+3*et**2*(
    806400+(1435*pi**2-430016)*eta+161140*eta**2-6300*eta**3)+70*et**4*(15*eta**3-1915*
    eta**2+11233*eta-12288)))
    return PNx(a0,a1,a2,a3,0,x,order)


def phiv(eta,et,u,x,order):
    v=vH(eta,et,u,x,order)
    a0=v
    a1=3*v/(et**2-1)
    a2=(-1/32*(8*v*(-78+28*eta+et**2*(-51+26*eta))+4*et**2*(3*eta**2-19*eta-1)*sin(2*v
    )+et**3*eta*(-1+3*eta)*sin(3*v))/(et**2-1)**2)

    a3=(1/26880/(et**2-1)**3*(et**2*(84000+1180064*eta-30135*pi**2*eta-442400*eta**2+
    10080*eta**3+280*et**2*(93*eta**3-781*eta**2+886*eta+24))*sin(2*v)+et**3*eta*(113208
    -4305*pi**2-101780*eta+7140*eta**2+35*et**2*(129*eta**2-137*eta+33))*sin(3*v)+210*v
    *(16*et**4*(65*eta**2-110*eta+156)+18240+4*(123*pi**2-6344)*eta+896*eta**2+et**2*(
    28128+3*(41*pi**2-9280)*eta+5120*eta**2))+140*et**4*eta*(15*eta**2-57*eta+82)*sin(4
    *v)+105*et**5*eta*(5*eta**2-5*eta+1)*sin(5*v)))
    return PNx(a0,a1,a2,a3,0,x,order)


def get_gw_ra_dec(a1,b1,e1,d1):
    psrra=     15*(a1+b1/60)*pi/180
    psrdec =   (e1+e1/np.abs(e1)*d1/60)*pi/180
    return psrra, psrdec


def odes(eta,b1,y,t):
    et=y[0]
    n=y[1]
    u=y[2]
    w=et*cosh(u)-1
    x=n**(2/3)
    
    dedt_Q=(8/15*(et**2-1)*x**4*(35*(1-et**2)+(49-9*et**2)*w+17*w**2+3*w**3)*eta/et/w**6)
    dndt_Q=(8/5*x**(11/2)*(35*(1-et**2)+49*w+32*w**2+6*w**3-9*w*et**2)*eta/w**6)
    
    dedt_1PN=(-2/315/et/w**8*x**5*(-17640*(et**2-1)**4+63*w*(et**2-1)**3*(140*eta+657)-105*w**2*
    (et**2-1)**2*(13+454*eta+9*et**2*(3+2*eta))-w**4*(et**2-1)*(36825-53060*eta+9*et**2*(
    -2169+560*eta))+6*w**6*(360-553*eta+et**2*(-444+637*eta))-28*w**3*(et**2-1)*(1827-
    2755*eta+et**2*(-1767+1105*eta)+w**5*(10215-18088*eta+et**2*(-12735+20608*eta)))))
    
    dndt_1PN=(2/35/w**8*x**(13/2)*(w**6*(180-588*eta)+w**5*(1340-5852*eta)+2*w**4*(9*et**2*
    (21*eta-1)-8589*eta+1003)+35*w**3*(et**2*(244*eta-5)-684*eta+21)+35*w**2*(et**2-1)*
    (9*et**2*(2*eta-17)+454*eta+193)-21*w*(et**2-1)**2*(140*eta+657)+5880*(et**2-1)**3))
    
    dudt_Q=1/w
    dudt_2PN=(1/8*(-1+et)*((1+et)/(-1+et))**(1/2)*n*(-60+24*eta+et*(-15+eta)*eta*cos(2*arctan(
    ((1+et)/(-1+et))**(1/2)*tanh(1/2*u))))/(et**2-1)**(1/2)/(1-et*cosh(u))**2/(-1+et*
    cosh(u)))

    dudt_25PN=(8*(1-et**2)/(15*w**7)*eta*sinh(u)*(24*cosh(u)-26*et-9*et**2*cosh(u)
            +8*et*cosh(u)**2+3*et**2*cosh(u)**3))

    dudt_3PN=(1/6720*(1-et)*n*cosh(1/2*u)**2*(-840*et*((1+et)/(-1+et))**(1/2)*(4-eta)*(60-24*
    eta-et*(-15+eta)*eta*cos(2*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*u))))*cosh(u)
    *(1-et*cosh(u))*sech(1/2*u)**2+35*((1+et)/(-1+et))**(1/2)*(8640+(123*pi**2-13184)*
    eta+960*eta**2+96*et**2*(11*eta**2-29*eta+30))*(-1+et*cosh(u))*sech(1/2*u)**2+et*((
    1+et)/(-1+et))**(1/2)*(67200-3*(1435*pi**2+105*et**2-47956)*eta-105*(135*et**2+592)
    *eta**2+35*(65*et**2-8)*eta**3)*cos(2*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*u)))*
    (-1+et*cosh(u))*sech(1/2*u)**2+840*et**2*((1+et)/(-1+et))**(1/2)*eta*(3*eta**2-49*
    eta+116)*cos(4*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*u)))*(-1+et*cosh(u))*sech
    (1/2*u)**2+105*et**3*((1+et)/(-1+et))**(1/2)*eta*(13*eta**2-73*eta+23)*cos(6*arctan
    (((1+et)/(-1+et))**(1/2)*tanh(1/2*u)))*(-1+et*cosh(u))*sech(1/2*u)**2-3360*et**2*(
    (1+et)/(-1+et))**(1/2)*(4-eta)*(60-24*eta-et*(-15+eta)*eta*cos(2*arctan(((1+et)/
    (-1+et))**(1/2)*tanh(1/2*u))))*sinh(1/2*u)**2-1680*et**2*(1+et)*(-15+eta)*(-4+eta)
    *eta*sin(2*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*u)))*tanh(1/2*u))/(et**2-1)**(3
    /2)/(1-et*cosh(u))**2/(-1+et*cosh(u))**2)

    dudt= n*(dudt_Q+x**2*dudt_2PN+x**(5/2)*dudt_25PN+x**3*dudt_3PN)

    return [(dedt_Q),(dndt_Q),dudt]
    #return [(dedt_Q+dedt_1PN),(dndt_Q+dndt_1PN),dldt]


def solve_rr(eta,b1,y0,Ti,Tf,Tarr):
    sol = solve_ivp(lambda t,y:odes(eta,b1,y,t),[Ti,Tf],y0,t_eval=Tarr,rtol=1e-10, atol=1e-10)
    Earr=sol.y[0]
    Narr=sol.y[1]
    Uarr=sol.y[2]
    return Earr, Narr, Uarr


def solve_rr2(eta,b1,y0,Ti,Tf,Tarr):
    sol =odeint(lambda t,y:odes(eta,b1,y,t),y0,Tarr,tfirst=True,rtol=1e-10, atol=1e-10)
    earr=sol[:, 0]
    narr=sol[:,1]
    Uarr=sol[:,2]
    return earr, narr, Uarr

def solve_rr3(eta,b1,y0,Ti,Tf):
    sol = solve_ivp(lambda t,y:odes(eta,b1,y,t),[Ti,Tf],y0,rtol=1e-10, atol=1e-10)
    Earr=sol.y[0]
    Narr=sol.y[1]
    Uarr=sol.y[2]
    Tarr=sol.t
    return Earr, Narr, Uarr, Tarr

def coeff(et,eta):
   
    a0=(et**2-1)**(1/2)
    a1=-1/6*(7*et**2*eta-6*et**2-eta)/(et**2-1)**(1/2)
    a2=(1/72*(365*et**4*eta**2-1539*et**4*eta+1242*et**4-58*et**2*eta**2-690*et**2*eta
        -792*et**2+17*eta**2+69*eta+738)/(et**2-1)**(3/2))
    a3=(369/64/(et**2-1)**(5/2)*((-68204/29889*et**6-15004/9963*et**4-1580/9963*
        et**2-412/29889)*eta**3+(34700/3321*et**6+27196/3321*et**4+3412/3321*et**2+17924/
        3321)*eta**2+(-19700/1107*et**6-4036/1107*et**4+(-1772228/38745+pi**2)*et**2+1/3*pi**
        2-13916/1107)*eta+5264/1107+14096/1107*et**6-1456/123*et**4+7312/369*et**2))
    return (a0,a1,a2,a3)
    

def cal_xQ(b1,a0,a1):
    b=b1
    return (a0/(b-a1))


def cal_x(x,b1,a0,a1,a2,a3,order):
    
    b=b1
    if order<=1:
        return a0/(b-a1)
    if order==2:
        return a0/(b-a1-a2*x)
    if order==3:
        return a0/(b-a1-a2*x-a3*x**2)


def get_x(et,eta,b1,order):
    
    b=b1
    a0, a1, a2, a3 =coeff(et,eta)

    x0=cal_xQ(b1,a0,a1)
    tol = 1e-15
    diff = 1
    x1 = x0
    step = 0
    while diff>tol:
        x2 = cal_x(x1,b1,a0,a1,a2,a3,order)
        diff = x2-x1
        x1 = x2
        step += 1
    return (x1, step)
    
    
def get_x_PN(et,eta,b):
    A1=(1/6*(-7*et**2*eta+6*et**2+eta)/b/(et**2-1)**(1/2))
    A2=(1/72*(463*et**4*eta**2-1707*et**4*eta+1314*et**4-86*et**2*eta**2-666*et**2*eta-792*et**
        2+19*eta**2+69*eta+738)/b**2/(et**2-1))
    A3=(1/181440/b**3/(et**2-1)**(3/2)*(5894560*et**6*eta**3-28004760*et**6*eta**2+41840820*et**6
        *eta+480480*et**4*eta**3-22891680*et**6-12063240*et**4*eta**2-1046115*pi**2*et**2*
        eta+389340*et**4*eta+406560*et**2*eta**3+18370800*et**4+259560*et**2*eta**2-348705*pi**2
        *eta+54835596*et**2*eta-7840*eta**3-26308800*et**2-5733000*eta**2+12220740*eta-
        4974480))
    FAC=sqrt(et**2-1)/b
    return FAC*(1+A1+A2+A3)


def get_b(et,eta,x):
    a0=coeff(M1,et,eta)[0]
    a1=coeff(M1,et,eta)[1]
    a2=coeff(M1,et,eta)[2]
    a3=coeff(M1,et,eta)[3]
    B=a0/x+a1+a2*x+a3*x**2
    return(B)
    


def get_u_hat(l,e):

 
     alpha = (e-1)/(4*e + 0.5)

     alpha3 = alpha*alpha*alpha

     beta   = (l/2)/(4*e + 0.5)

     beta2  = beta*beta;

     z = np.zeros_like(l)

     z= np.cbrt(beta + np.sqrt(alpha3 + beta2))
     
     s=(z - alpha/z)
     s5= s*s*s*s*s
     
     ds= 0.071*s5/((1+0.45*s*s)*(1+4*s*s)*e)
     w= s+ds
        
     u = 3*np.log(w+np.sqrt(1+w*w))
     
     esu= e*np.sinh(u)
     ecu= e*np.cosh(u)
     
     fu  = -u + esu - l
     f1u = -1 + ecu  
     f2u = esu
     f3u = ecu
     f4u = esu
     f5u = ecu

     u1 = -fu/ f1u
     u2 = -fu/(f1u + f2u*u1/2)
     u3 = -fu/(f1u + f2u*u2/2 + f3u*(u2*u2)/6.0)
     u4 = -fu/(f1u + f2u*u3/2 + f3u*(u3*u3)/6.0 + f4u*(u3*u3*u3)/24.0)
     u5 = -fu/(f1u + f2u*u4/2 + f3u*(u4*u4)/6.0 + f4u*(u4*u4*u4)/24.0 + f5u*(u4*u4*u4*u4)/120.0)
     uM = (u + u5)
 
     return uM


def get_u(l,et,eta,b1,order):
    
    U=get_u_hat(l,et)
    x=get_x(et,eta,b1,order)[0]

    a0=U

    a2=(1/8*x**2*(24*(-5+2*eta)*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*U))*(-1+et*cosh(U
        ))/(et**2-1)**(1/2)+et*(-15+eta)*eta*sinh(U))/(-1+et*cosh(U))**2)


    a3=(x**3*(1/8*et*(-4+eta)*(-60+3*(5*et**2+8)*eta-et**2*eta**2+et*(eta**2-39*eta+60)*cosh
        (U))*sinh(U)/(et**2-1)/(-1+et*cosh(U))**3-1/6720/(et**2-1)**(3/2)/(-1+et*cosh(U))*(
        et*(et**2-1)**(1/2)*(67200-3*(1435*pi**2+105*et**2-47956)*eta-105*(135*et**2+592)*
        eta**2+35*(65*et**2-8)*eta**3)*sinh(U)/(-1+et*cosh(U))+70*(8640+(123*pi**2-13184)*
        eta+960*eta**2+96*et**2*(11*eta**2-29*eta+30))*arctan(((1+et)/(-1+et))**(1/2)*tanh(
        1/2*U))+840*et**2*(et**2-1)**(1/2)*eta*(3*eta**2-49*eta+116)*(et-cosh(U))*sinh(U)/(
        -1+et*cosh(U))**2-35/2*et**3*(et**2-1)**(1/2)*eta*(13*eta**2-73*eta+23)*(-7*et**2-2+
        12*et*cosh(U)+(et**2-4)*cosh(2*U))*sinh(U)/(-1+et*cosh(U))**3)))

    if order<=1:
        return a0
    if order==2:
        return a0+a2
    if order==3:
        return a0+a2+a3


def get_u_v2(l,et,eta,x,order):
    
    U=get_u_hat(l,et)
    
    a0=U

    a2=(1/8*x**2*(24*(-5+2*eta)*arctan(((1+et)/(-1+et))**(1/2)*tanh(1/2*U))*(-1+et*cosh(U
        ))/(et**2-1)**(1/2)+et*(-15+eta)*eta*sinh(U))/(-1+et*cosh(U))**2)


    a3=(x**3*(1/8*et*(-4+eta)*(-60+3*(5*et**2+8)*eta-et**2*eta**2+et*(eta**2-39*eta+60)*cosh
        (U))*sinh(U)/(et**2-1)/(-1+et*cosh(U))**3-1/6720/(et**2-1)**(3/2)/(-1+et*cosh(U))*(
        et*(et**2-1)**(1/2)*(67200-3*(1435*pi**2+105*et**2-47956)*eta-105*(135*et**2+592)*
        eta**2+35*(65*et**2-8)*eta**3)*sinh(U)/(-1+et*cosh(U))+70*(8640+(123*pi**2-13184)*
        eta+960*eta**2+96*et**2*(11*eta**2-29*eta+30))*arctan(((1+et)/(-1+et))**(1/2)*tanh(
        1/2*U))+840*et**2*(et**2-1)**(1/2)*eta*(3*eta**2-49*eta+116)*(et-cosh(U))*sinh(U)/(
        -1+et*cosh(U))**2-35/2*et**3*(et**2-1)**(1/2)*eta*(13*eta**2-73*eta+23)*(-7*et**2-2+
        12*et*cosh(U)+(et**2-4)*cosh(2*U))*sinh(U)/(-1+et*cosh(U))**3)))

    if order<=1:
        return a0
    if order==2:
        return a0+a2
    if order==3:
        return a0+a2+a3
    
def Fomg(eta,b1,e,omg):
    if type(omg) is np.ndarray:
        return np.array([Fomg(eta,b1,e,omg1) for omg1 in omg])
    b=b1
    a=b/np.sqrt(e**2-1)
    u=omg*e*a**(3/2)
    p=1j*u/e
    bk1 = besselk(p+1,u)
    bk0 = besselk(p,u)
    fac=32/5*(eta/a)**2*(p/u**2)**2*np.exp(-1j*np.pi*p)
    sed=(u**2*(p**2+u**2+1)*(p**2+u**2)*bk1**2
    -2*u*((p-3/2)*u**2+p*(p-1)**2)*(p**2+u**2)*bk0*bk1
    +2*(u**6/2+(2*p**2-3/2*p+1/6)*u**4+(5/2*p**4-7/2*p**3+p**2)*u**2+p**4*(p-1)**2)*bk0**2  )
    return float(np.abs(fac*sed))

def get_max(eta,b1,e1):
    x0=get_x(e1,eta,b1,0)[0]
    n=x0**(3/2)
    val=optimize.fmin(lambda omg:-Fomg(eta,b1,e1,omg),n,  xtol=1e-10, ftol=1e-10,disp= False)[0]
    return val