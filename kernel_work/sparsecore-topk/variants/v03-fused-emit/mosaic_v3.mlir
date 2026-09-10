jaxpr for sc_vector_subcore
let jaxpr = { lambda ; a:i32[] b:Ref<vmem>{i32[1024]} c:Ref<hbm>{i32[10485760]} d:Ref<vmem>{i32[10240]}
    e:Ref<semaphore_mem>{dma_sem[4]} f:Ref<vmem>{i32[4112]} g:i32[16] h:Ref<vmem>{i32[272]}
    i:Ref<vmem>{i32[2064]} j:Ref<vmem>{i32[10256]} k:Ref<vmem>{i32[10256]} l:Ref<hbm>{i32[2097152]}
    m:i32[] n:i32[]. let
    o:i32[] = add m 1:i32[]
    p:i32[] = mul m 32:i32[]
    q:i32[] = convert_element_type[new_dtype=int32 weak_type=False] p
    r:i32[] = add q a
    s:bool[] = lt r 1024:i32[]
    t:i32[] = min r 1023:i32[]
    u:i32[16] <- b[t:t+16]
    v:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] u
    w:i32[] = squeeze[dimensions=(0,)] v
    x:i32[] = min w 10240:i32[]
    y:i32[] = jit[name=_where jaxpr=_where] s x 0:i32[]
    z:i32[] = mul t 10240:i32[]
    ba:bool[] = gt y 0:i32[]
    bb:i32[] = min 10240:i32[] y
    bc:i32[] = jit[name=_where jaxpr=_where] ba bb 0:i32[]
    bd:i32[] = add bc 15:i32[]
    be:i32[] = shift_right_arithmetic bd 4:i32[]
    bf:bool[] = gt bc 0:i32[]
    bg:i32[] = convert_element_type[new_dtype=int32 weak_type=False] bf
    cond[
      branches=(
        { lambda ; bh:Ref<hbm>{i32[10485760]} bi:i32[] bj:Ref<vmem>{i32[10240]} bk:Ref<semaphore_mem>{dma_sem[4]}
            bl:i32[]. let
            
          in () }
        { lambda ; bm:Ref<hbm>{i32[10485760]} bn:i32[] bo:Ref<vmem>{i32[10240]} bp:Ref<semaphore_mem>{dma_sem[4]}
            bq:i32[]. let
            dma_start(p0) bm[bn:bn+10240] -> bo[...] bp[1]
            dma_wait bo[...] bp[1]
            parallel_loop[
              jaxpr={ lambda br:Ref<vmem>{i32[10240]}; bs:i32[]. let
                  bt:i32[] = mul bs 16:i32[]
                  bu:i32[16] <- br[bt:bt+16]
                  bv:i32[16] = shift_right_arithmetic bu 31:i32[]
                  bw:i32[16] = and bv 2147483647:i32[]
                  bx:i32[16] = xor bu bw
                  by:i32[] = mul bs 16:i32[]
                  bz:i32[16], br[by:by+16] <- br[by:by+16], bx
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] bq 1:i32[] bo
          in () }
      )
    ] bg c z d e be
    ca:bool[] = gt bc 0:i32[]
    cb:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ca
    cond[
      branches=(
        { lambda ; cc:Ref<vmem>{i32[4112]} cd:i32[] ce:Ref<vmem>{i32[10240]} cf:i32[16]
            cg:i32[] ch:Ref<vmem>{i32[272]}. let
            
          in () }
        { lambda ; ci:Ref<vmem>{i32[4112]} cj:i32[] ck:Ref<vmem>{i32[10240]} cl:i32[16]
            cm:i32[] cn:Ref<vmem>{i32[272]}. let
            parallel_loop[
              jaxpr={ lambda co:Ref<vmem>{i32[4112]}; cp:i32[]. let
                  cq:i32[16] = broadcast_in_dim 0:i32[]
                  cr:i32[] = mul cp 16:i32[]
                  cs:i32[16], co[cr:cr+16] <- co[cr:cr+16], cq
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] 256:i32[] 1:i32[] ci
            parallel_loop[
              jaxpr=jaxpr9
              tree=PyTreeDef((*, *, *, [*, *, *, *], []))
              unroll=4
            ] 0:i32[] cj 1:i32[] ck cl cm ci
            parallel_loop[
              jaxpr=jaxpr10
              tree=PyTreeDef((*, *, *, [*, *], []))
              unroll=2
            ] 0:i32[] 16:i32[] 1:i32[] ci cn
          in () }
      )
    ] cb f be d g bc h
    ct:bool[] = eq bc 0:i32[]
    cu:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ct
    cond[
      branches=(
        { lambda ; cv:Ref<vmem>{i32[272]}. let  in () }
        { lambda ; cw:Ref<vmem>{i32[272]}. let
            parallel_loop[
              jaxpr={ lambda cx:Ref<vmem>{i32[272]}; cy:i32[]. let
                  cz:i32[16] = broadcast_in_dim 0:i32[]
                  da:i32[] = mul cy 16:i32[]
                  db:i32[16], cx[da:da+16] <- cx[da:da+16], cz
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] 16:i32[] 1:i32[] cw
          in () }
      )
    ] cu h
    dc:i32[] = parallel_loop[
      jaxpr={ lambda dd:Ref<vmem>{i32[272]}; de:i32[] df:i32[]. let
          dg:i32[] = mul de 16:i32[]
          dh:i32[16] <- dd[dg:dg+16]
          di:bool[16] = broadcast_in_dim True:bool[]
          dj:i32[16] = masked_cumsum dh di
          dk:i32[16] = add dj df
          dl:i32[] = mul de 16:i32[]
          dm:i32[16], dd[dl:dl+16] <- dd[dl:dl+16], dk
          dn:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] dk
          do:i32[] = squeeze[dimensions=(0,)] dn
        in (do,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] 16:i32[] 1:i32[] h 0:i32[]
    dp:bool[] = le dc 2048:i32[]
    dq:i32[] = sub dc 2048:i32[]
    dr:i32[] = parallel_loop[
      jaxpr={ lambda ds:Ref<vmem>{i32[272]} dt:i32[]; du:i32[] dv:i32[]. let
          dw:i32[] = mul du 16:i32[]
          dx:i32[16] <- ds[dw:dw+16]
          dy:bool[16] = le dx dt
          dz:i32[16] = all_reduce_population_count[reduce=1] dy
          ea:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] dz
          eb:i32[] = squeeze[dimensions=(0,)] ea
          ec:i32[] = add dv eb
        in (ec,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] 16:i32[] 1:i32[] h dq 0:i32[]
    ed:i32[16] <- h[dr:dr+16]
    ee:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ed
    ef:i32[] = squeeze[dimensions=(0,)] ee
    eg:bool[] = gt dr 0:i32[]
    eh:i32[] = sub dr 1:i32[]
    ei:i32[] = max eh 0:i32[]
    ej:i32[16] <- h[ei:ei+16]
    ek:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ej
    el:i32[] = squeeze[dimensions=(0,)] ek
    em:i32[] = jit[name=_where jaxpr=_where] eg el 0:i32[]
    en:i32[] = sub dc ef
    eo:i32[] = sub ef em
    ep:i32[] = sub 2048:i32[] en
    eq:i32[] = sub dr 128:i32[]
    er:bool[] = eq ep eo
    es:bool[] = or dp er
    et:bool[] = lt dc 2048:i32[]
    eu:bool[] = and s et
    ev:i32[] = jit[name=_where jaxpr=_where1] eu 128:i32[] 0:i32[]
    parallel_loop[
      jaxpr={ lambda ew:Ref<vmem>{i32[2064]}; ex:i32[]. let
          ey:i32[16] = broadcast_in_dim -1:i32[]
          ez:i32[] = mul ex 16:i32[]
          fa:i32[16], ew[ez:ez+16] <- ew[ez:ez+16], ey
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] ev 1:i32[] i
    fb:i32[] = jit[name=_where jaxpr=_where] s be 0:i32[]
    fc:i32[] fd:i32[] = parallel_loop[
      jaxpr=jaxpr1
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] fb 1:i32[] d g bc eq es dp i j k 0:i32[] 0:i32[]
    fe:i32[] = add eo 15:i32[]
    ff:i32[] = shift_right_arithmetic fe 4:i32[]
    fg:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda fh:Ref<vmem>{i32[4112]}; fi:i32[]. let
          fj:i32[16] = broadcast_in_dim 0:i32[]
          fk:i32[] = mul fi 16:i32[]
          fl:i32[16], fh[fk:fk+16] <- fh[fk:fk+16], fj
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] fg 1:i32[] f
    fm:i32[] = jit[name=_where jaxpr=_where2] es 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr2
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] fm 1:i32[] j g eo 24:i32[] eq f
    fn:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr3 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      fn 1:i32[] f h
    fo:i32[] = parallel_loop[
      jaxpr={ lambda fp:Ref<vmem>{i32[272]}; fq:i32[] fr:i32[]. let
          fs:i32[] = mul fq 16:i32[]
          ft:i32[16] <- fp[fs:fs+16]
          fu:bool[16] = broadcast_in_dim True:bool[]
          fv:i32[16] = masked_cumsum ft fu
          fw:i32[16] = add fv fr
          fx:i32[] = mul fq 16:i32[]
          fy:i32[16], fp[fx:fx+16] <- fp[fx:fx+16], fw
          fz:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] fw
          ga:i32[] = squeeze[dimensions=(0,)] fz
        in (ga,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] fn 1:i32[] h 0:i32[]
    gb:i32[] = sub fo ep
    gc:i32[] = parallel_loop[
      jaxpr={ lambda gd:Ref<vmem>{i32[272]} ge:i32[]; gf:i32[] gg:i32[]. let
          gh:i32[] = mul gf 16:i32[]
          gi:i32[16] <- gd[gh:gh+16]
          gj:bool[16] = le gi ge
          gk:i32[16] = all_reduce_population_count[reduce=1] gj
          gl:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gk
          gm:i32[] = squeeze[dimensions=(0,)] gl
          gn:i32[] = add gg gm
        in (gn,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] fn 1:i32[] h gb 0:i32[]
    go:i32[16] <- h[gc:gc+16]
    gp:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] go
    gq:i32[] = squeeze[dimensions=(0,)] gp
    gr:bool[] = gt gc 0:i32[]
    gs:i32[] = sub gc 1:i32[]
    gt:i32[] = max gs 0:i32[]
    gu:i32[16] <- h[gt:gt+16]
    gv:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gu
    gw:i32[] = squeeze[dimensions=(0,)] gv
    gx:i32[] = jit[name=_where jaxpr=_where] gr gw 0:i32[]
    gy:i32[] = sub fo gq
    gz:i32[] = sub gq gx
    ha:i32[] = sub ep gy
    hb:i32[] = sub 24:i32[] 16:i32[]
    hc:i32[] = shift_left eq hb
    hd:i32[] = or hc gc
    he:bool[] = eq ha gz
    hf:bool[] = or es he
    hg:bool[] = eq ha 0:i32[]
    hh:bool[] = or hf hg
    hi:i32[] = jit[name=_where jaxpr=_where3] es eq hd
    hj:i32[] = jit[name=_where jaxpr=_where3] es 24:i32[] 16:i32[]
    hk:i32[] = add en gy
    hl:i32[] = jit[name=_where jaxpr=_where3] es en hk
    hm:i32[] = jit[name=_where jaxpr=_where3] es ep ha
    hn:i32[] = jit[name=_where jaxpr=_where3] es eo gz
    ho:i32[] = jit[name=_where jaxpr=_where1] hh 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda hp:Ref<vmem>{i32[4112]}; hq:i32[]. let
          hr:i32[16] = broadcast_in_dim 0:i32[]
          hs:i32[] = mul hq 16:i32[]
          ht:i32[16], hp[hs:hs+16] <- hp[hs:hs+16], hr
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] ho 1:i32[] f
    hu:i32[] = jit[name=_where jaxpr=_where2] hh 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr4
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] hu 1:i32[] j g hn hj hi f
    hv:i32[] = jit[name=_where jaxpr=_where1] hh 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr5 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      hv 1:i32[] f h
    hw:i32[] = parallel_loop[
      jaxpr={ lambda hx:Ref<vmem>{i32[272]}; hy:i32[] hz:i32[]. let
          ia:i32[] = mul hy 16:i32[]
          ib:i32[16] <- hx[ia:ia+16]
          ic:bool[16] = broadcast_in_dim True:bool[]
          id:i32[16] = masked_cumsum ib ic
          ie:i32[16] = add id hz
          if:i32[] = mul hy 16:i32[]
          ig:i32[16], hx[if:if+16] <- hx[if:if+16], ie
          ih:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] ie
          ii:i32[] = squeeze[dimensions=(0,)] ih
        in (ii,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] hv 1:i32[] h 0:i32[]
    ij:i32[] = sub hw hm
    ik:i32[] = parallel_loop[
      jaxpr={ lambda il:Ref<vmem>{i32[272]} im:i32[]; in:i32[] io:i32[]. let
          ip:i32[] = mul in 16:i32[]
          iq:i32[16] <- il[ip:ip+16]
          ir:bool[16] = le iq im
          is:i32[16] = all_reduce_population_count[reduce=1] ir
          it:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] is
          iu:i32[] = squeeze[dimensions=(0,)] it
          iv:i32[] = add io iu
        in (iv,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] hv 1:i32[] h ij 0:i32[]
    iw:i32[16] <- h[ik:ik+16]
    ix:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] iw
    iy:i32[] = squeeze[dimensions=(0,)] ix
    iz:bool[] = gt ik 0:i32[]
    ja:i32[] = sub ik 1:i32[]
    jb:i32[] = max ja 0:i32[]
    jc:i32[16] <- h[jb:jb+16]
    jd:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] jc
    je:i32[] = squeeze[dimensions=(0,)] jd
    jf:i32[] = jit[name=_where jaxpr=_where] iz je 0:i32[]
    jg:i32[] = sub hw iy
    jh:i32[] = sub iy jf
    ji:i32[] = sub hm jg
    jj:i32[] = sub hj 8:i32[]
    jk:i32[] = shift_left hi jj
    jl:i32[] = or jk ik
    jm:bool[] = eq ji jh
    jn:bool[] = or hh jm
    jo:bool[] = eq ji 0:i32[]
    jp:bool[] = or jn jo
    jq:i32[] = jit[name=_where jaxpr=_where3] hh hi jl
    jr:i32[] = jit[name=_where jaxpr=_where3] hh hj 8:i32[]
    js:i32[] = add hl jg
    jt:i32[] = jit[name=_where jaxpr=_where3] hh hl js
    ju:i32[] = jit[name=_where jaxpr=_where3] hh hm ji
    jv:i32[] = jit[name=_where jaxpr=_where3] hh hn jh
    jw:i32[] = jit[name=_where jaxpr=_where1] jp 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda jx:Ref<vmem>{i32[4112]}; jy:i32[]. let
          jz:i32[16] = broadcast_in_dim 0:i32[]
          ka:i32[] = mul jy 16:i32[]
          kb:i32[16], jx[ka:ka+16] <- jx[ka:ka+16], jz
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] jw 1:i32[] f
    kc:i32[] = jit[name=_where jaxpr=_where2] jp 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr6
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] kc 1:i32[] j g jv jr jq f
    kd:i32[] = jit[name=_where jaxpr=_where1] jp 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr7 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      kd 1:i32[] f h
    ke:i32[] = parallel_loop[
      jaxpr={ lambda kf:Ref<vmem>{i32[272]}; kg:i32[] kh:i32[]. let
          ki:i32[] = mul kg 16:i32[]
          kj:i32[16] <- kf[ki:ki+16]
          kk:bool[16] = broadcast_in_dim True:bool[]
          kl:i32[16] = masked_cumsum kj kk
          km:i32[16] = add kl kh
          kn:i32[] = mul kg 16:i32[]
          ko:i32[16], kf[kn:kn+16] <- kf[kn:kn+16], km
          kp:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] km
          kq:i32[] = squeeze[dimensions=(0,)] kp
        in (kq,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] kd 1:i32[] h 0:i32[]
    kr:i32[] = sub ke ju
    ks:i32[] = parallel_loop[
      jaxpr={ lambda kt:Ref<vmem>{i32[272]} ku:i32[]; kv:i32[] kw:i32[]. let
          kx:i32[] = mul kv 16:i32[]
          ky:i32[16] <- kt[kx:kx+16]
          kz:bool[16] = le ky ku
          la:i32[16] = all_reduce_population_count[reduce=1] kz
          lb:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] la
          lc:i32[] = squeeze[dimensions=(0,)] lb
          ld:i32[] = add kw lc
        in (ld,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] kd 1:i32[] h kr 0:i32[]
    le:i32[16] <- h[ks:ks+16]
    lf:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] le
    lg:i32[] = squeeze[dimensions=(0,)] lf
    lh:bool[] = gt ks 0:i32[]
    li:i32[] = sub ks 1:i32[]
    lj:i32[] = max li 0:i32[]
    lk:i32[16] <- h[lj:lj+16]
    ll:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] lk
    lm:i32[] = squeeze[dimensions=(0,)] ll
    ln:i32[] = jit[name=_where jaxpr=_where] lh lm 0:i32[]
    lo:i32[] = sub ke lg
    lp:i32[] = sub lg ln
    lq:i32[] = sub ju lo
    lr:i32[] = sub jr 0:i32[]
    ls:i32[] = shift_left jq lr
    lt:i32[] = or ls ks
    lu:bool[] = eq lq lp
    lv:bool[] = or jp lu
    lw:bool[] = eq lq 0:i32[]
    lx:bool[] = or lv lw
    ly:i32[] = jit[name=_where jaxpr=_where3] jp jq lt
    lz:i32[] = jit[name=_where jaxpr=_where3] jp jr 0:i32[]
    ma:i32[] = add jt lo
    mb:i32[] = jit[name=_where jaxpr=_where3] jp jt ma
    mc:i32[] = jit[name=_where jaxpr=_where3] jp ju lq
    md:i32[] = jit[name=_where jaxpr=_where3] jp jv lp
    me:i32[] = jit[name=_where jaxpr=_where2] dp 0:i32[] mc
    mf:bool[] = not es
    mg:bool[] = and s mf
    mh:i32[] = jit[name=_where jaxpr=_where] mg ff 0:i32[]
    mi:i32[] mj:i32[] = parallel_loop[
      jaxpr=jaxpr8
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] mh 1:i32[] j g md k lz ly me i fc 0:i32[]
    mk:i32[] = convert_element_type[new_dtype=int32 weak_type=False] s
    cond[
      branches=(
        { lambda ; ml:i32[] mm:Ref<vmem>{i32[2064]} mn:Ref<hbm>{i32[2097152]} mo:Ref<semaphore_mem>{dma_sem[4]}. let
            
          in () }
        { lambda ; mp:i32[] mq:Ref<vmem>{i32[2064]} mr:Ref<hbm>{i32[2097152]} ms:Ref<semaphore_mem>{dma_sem[4]}. let
            mt:i32[] = mul mp 2048:i32[]
            dma_start(p0) mq[:2048] -> mr[mt:mt+2048] ms[2]
            dma_wait mr[mt:mt+2048] ms[2]
          in () }
      )
    ] mk t i l e
  in (o, 0:i32[]) } in
let _where = { lambda ; mu:bool[] mv:i32[] mw:i32[]. let
    mx:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mw
    my:i32[] = select_n mu mx mv
  in (my,) } in
let _where1 = { lambda ; mz:bool[] na:i32[] nb:i32[]. let
    nc:i32[] = select_n mz nb na
  in (nc,) } in
let jaxpr1 = { lambda nd:Ref<vmem>{i32[10240]} ne:i32[16] nf:i32[] ng:i32[] nh:bool[]
    ni:bool[] nj:Ref<vmem>{i32[2064]} nk:Ref<vmem>{i32[10256]} nl:Ref<vmem>{i32[10256]}; nm:i32[]
    nn:i32[] no:i32[]. let
    np:i32[] = mul nm 16:i32[]
    nq:i32[16] <- nd[np:np+16]
    nr:i32[] = mul nm 16:i32[]
    ns:i32[16] = add nr ne
    nt:bool[16] = lt ns nf
    nu:bool[16] = gt nq -2139095041:i32[]
    nv:bool[16] = and nt nu
    nw:i32[16] = shift_right_arithmetic nq 24:i32[]
    nx:bool[16] = gt nw ng
    ny:bool[16] = and nv nx
    nz:bool[16] = eq nw ng
    oa:bool[16] = and nv nz
    ob:bool[16] = or ny oa
    oc:bool[16] = jit[name=_where jaxpr=_where4] nh ob ny
    od:bool[16] = jit[name=_where jaxpr=_where4] ni nv oc
    oe:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] nj ns nn od
    of:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (10256,), ())], [*]),))
    ] nk nq no oa
    og:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (10256,), ())], [*]),))
    ] nl ns no oa
    oh:i32[16] = all_reduce_population_count[reduce=1] od
    oi:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] oh
    oj:i32[] = squeeze[dimensions=(0,)] oi
    ok:i32[] = add nn oj
    ol:i32[16] = all_reduce_population_count[reduce=1] oa
    om:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ol
    on:i32[] = squeeze[dimensions=(0,)] om
    oo:i32[] = add no on
  in (ok, oo) } in
let _where2 = { lambda ; op:bool[] oq:i32[] or:i32[]. let
    os:i32[] = convert_element_type[new_dtype=int32 weak_type=False] oq
    ot:i32[] = select_n op or os
  in (ot,) } in
let jaxpr2 = { lambda ou:Ref<vmem>{i32[10256]} ov:i32[16] ow:i32[] ox:i32[] oy:i32[]
    oz:Ref<vmem>{i32[4112]}; pa:i32[]. let
    pb:i32[] = mul pa 16:i32[]
    pc:i32[16] <- ou[pb:pb+16]
    pd:i32[] = mul pa 16:i32[]
    pe:i32[16] = add pd ov
    pf:bool[16] = lt pe ow
    pg:i32[16] = shift_right_arithmetic pc ox
    ph:bool[16] = eq pg oy
    pi:bool[16] = and pf ph
    pj:i32[16] = shift_right_arithmetic pc 16:i32[]
    pk:i32[16] = and pj 255:i32[]
    pl:i32[16] = mul ov 256:i32[]
    pm:i32[16] = add pl pk
    pn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] oz pm pn pi
  in () } in
let jaxpr3 = { lambda po:Ref<vmem>{i32[4112]} pp:Ref<vmem>{i32[272]}; pq:i32[]. let
    pr:i32[] = mul pq 16:i32[]
    ps:i32[16] = broadcast_in_dim 0:i32[]
    pt:i32[] = add 0:i32[] pr
    pu:i32[16] <- po[pt:pt+16]
    pv:i32[16] = add ps pu
    pw:i32[] = add 256:i32[] pr
    px:i32[16] <- po[pw:pw+16]
    py:i32[16] = add pv px
    pz:i32[] = add 512:i32[] pr
    qa:i32[16] <- po[pz:pz+16]
    qb:i32[16] = add py qa
    qc:i32[] = add 768:i32[] pr
    qd:i32[16] <- po[qc:qc+16]
    qe:i32[16] = add qb qd
    qf:i32[] = add 1024:i32[] pr
    qg:i32[16] <- po[qf:qf+16]
    qh:i32[16] = add qe qg
    qi:i32[] = add 1280:i32[] pr
    qj:i32[16] <- po[qi:qi+16]
    qk:i32[16] = add qh qj
    ql:i32[] = add 1536:i32[] pr
    qm:i32[16] <- po[ql:ql+16]
    qn:i32[16] = add qk qm
    qo:i32[] = add 1792:i32[] pr
    qp:i32[16] <- po[qo:qo+16]
    qq:i32[16] = add qn qp
    qr:i32[] = add 2048:i32[] pr
    qs:i32[16] <- po[qr:qr+16]
    qt:i32[16] = add qq qs
    qu:i32[] = add 2304:i32[] pr
    qv:i32[16] <- po[qu:qu+16]
    qw:i32[16] = add qt qv
    qx:i32[] = add 2560:i32[] pr
    qy:i32[16] <- po[qx:qx+16]
    qz:i32[16] = add qw qy
    ra:i32[] = add 2816:i32[] pr
    rb:i32[16] <- po[ra:ra+16]
    rc:i32[16] = add qz rb
    rd:i32[] = add 3072:i32[] pr
    re:i32[16] <- po[rd:rd+16]
    rf:i32[16] = add rc re
    rg:i32[] = add 3328:i32[] pr
    rh:i32[16] <- po[rg:rg+16]
    ri:i32[16] = add rf rh
    rj:i32[] = add 3584:i32[] pr
    rk:i32[16] <- po[rj:rj+16]
    rl:i32[16] = add ri rk
    rm:i32[] = add 3840:i32[] pr
    rn:i32[16] <- po[rm:rm+16]
    ro:i32[16] = add rl rn
    rp:i32[16], pp[pr:pr+16] <- pp[pr:pr+16], ro
  in () } in
let _where3 = { lambda ; rq:bool[] rr:i32[] rs:i32[]. let
    rt:i32[] = select_n rq rs rr
  in (rt,) } in
let jaxpr4 = { lambda ru:Ref<vmem>{i32[10256]} rv:i32[16] rw:i32[] rx:i32[] ry:i32[]
    rz:Ref<vmem>{i32[4112]}; sa:i32[]. let
    sb:i32[] = mul sa 16:i32[]
    sc:i32[16] <- ru[sb:sb+16]
    sd:i32[] = mul sa 16:i32[]
    se:i32[16] = add sd rv
    sf:bool[16] = lt se rw
    sg:i32[16] = shift_right_arithmetic sc rx
    sh:bool[16] = eq sg ry
    si:bool[16] = and sf sh
    sj:i32[16] = shift_right_arithmetic sc 8:i32[]
    sk:i32[16] = and sj 255:i32[]
    sl:i32[16] = mul rv 256:i32[]
    sm:i32[16] = add sl sk
    sn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] rz sm sn si
  in () } in
let jaxpr5 = { lambda so:Ref<vmem>{i32[4112]} sp:Ref<vmem>{i32[272]}; sq:i32[]. let
    sr:i32[] = mul sq 16:i32[]
    ss:i32[16] = broadcast_in_dim 0:i32[]
    st:i32[] = add 0:i32[] sr
    su:i32[16] <- so[st:st+16]
    sv:i32[16] = add ss su
    sw:i32[] = add 256:i32[] sr
    sx:i32[16] <- so[sw:sw+16]
    sy:i32[16] = add sv sx
    sz:i32[] = add 512:i32[] sr
    ta:i32[16] <- so[sz:sz+16]
    tb:i32[16] = add sy ta
    tc:i32[] = add 768:i32[] sr
    td:i32[16] <- so[tc:tc+16]
    te:i32[16] = add tb td
    tf:i32[] = add 1024:i32[] sr
    tg:i32[16] <- so[tf:tf+16]
    th:i32[16] = add te tg
    ti:i32[] = add 1280:i32[] sr
    tj:i32[16] <- so[ti:ti+16]
    tk:i32[16] = add th tj
    tl:i32[] = add 1536:i32[] sr
    tm:i32[16] <- so[tl:tl+16]
    tn:i32[16] = add tk tm
    to:i32[] = add 1792:i32[] sr
    tp:i32[16] <- so[to:to+16]
    tq:i32[16] = add tn tp
    tr:i32[] = add 2048:i32[] sr
    ts:i32[16] <- so[tr:tr+16]
    tt:i32[16] = add tq ts
    tu:i32[] = add 2304:i32[] sr
    tv:i32[16] <- so[tu:tu+16]
    tw:i32[16] = add tt tv
    tx:i32[] = add 2560:i32[] sr
    ty:i32[16] <- so[tx:tx+16]
    tz:i32[16] = add tw ty
    ua:i32[] = add 2816:i32[] sr
    ub:i32[16] <- so[ua:ua+16]
    uc:i32[16] = add tz ub
    ud:i32[] = add 3072:i32[] sr
    ue:i32[16] <- so[ud:ud+16]
    uf:i32[16] = add uc ue
    ug:i32[] = add 3328:i32[] sr
    uh:i32[16] <- so[ug:ug+16]
    ui:i32[16] = add uf uh
    uj:i32[] = add 3584:i32[] sr
    uk:i32[16] <- so[uj:uj+16]
    ul:i32[16] = add ui uk
    um:i32[] = add 3840:i32[] sr
    un:i32[16] <- so[um:um+16]
    uo:i32[16] = add ul un
    up:i32[16], sp[sr:sr+16] <- sp[sr:sr+16], uo
  in () } in
let jaxpr6 = { lambda uq:Ref<vmem>{i32[10256]} ur:i32[16] us:i32[] ut:i32[] uu:i32[]
    uv:Ref<vmem>{i32[4112]}; uw:i32[]. let
    ux:i32[] = mul uw 16:i32[]
    uy:i32[16] <- uq[ux:ux+16]
    uz:i32[] = mul uw 16:i32[]
    va:i32[16] = add uz ur
    vb:bool[16] = lt va us
    vc:i32[16] = shift_right_arithmetic uy ut
    vd:bool[16] = eq vc uu
    ve:bool[16] = and vb vd
    vf:i32[16] = shift_right_arithmetic uy 0:i32[]
    vg:i32[16] = and vf 255:i32[]
    vh:i32[16] = mul ur 256:i32[]
    vi:i32[16] = add vh vg
    vj:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] uv vi vj ve
  in () } in
let jaxpr7 = { lambda vk:Ref<vmem>{i32[4112]} vl:Ref<vmem>{i32[272]}; vm:i32[]. let
    vn:i32[] = mul vm 16:i32[]
    vo:i32[16] = broadcast_in_dim 0:i32[]
    vp:i32[] = add 0:i32[] vn
    vq:i32[16] <- vk[vp:vp+16]
    vr:i32[16] = add vo vq
    vs:i32[] = add 256:i32[] vn
    vt:i32[16] <- vk[vs:vs+16]
    vu:i32[16] = add vr vt
    vv:i32[] = add 512:i32[] vn
    vw:i32[16] <- vk[vv:vv+16]
    vx:i32[16] = add vu vw
    vy:i32[] = add 768:i32[] vn
    vz:i32[16] <- vk[vy:vy+16]
    wa:i32[16] = add vx vz
    wb:i32[] = add 1024:i32[] vn
    wc:i32[16] <- vk[wb:wb+16]
    wd:i32[16] = add wa wc
    we:i32[] = add 1280:i32[] vn
    wf:i32[16] <- vk[we:we+16]
    wg:i32[16] = add wd wf
    wh:i32[] = add 1536:i32[] vn
    wi:i32[16] <- vk[wh:wh+16]
    wj:i32[16] = add wg wi
    wk:i32[] = add 1792:i32[] vn
    wl:i32[16] <- vk[wk:wk+16]
    wm:i32[16] = add wj wl
    wn:i32[] = add 2048:i32[] vn
    wo:i32[16] <- vk[wn:wn+16]
    wp:i32[16] = add wm wo
    wq:i32[] = add 2304:i32[] vn
    wr:i32[16] <- vk[wq:wq+16]
    ws:i32[16] = add wp wr
    wt:i32[] = add 2560:i32[] vn
    wu:i32[16] <- vk[wt:wt+16]
    wv:i32[16] = add ws wu
    ww:i32[] = add 2816:i32[] vn
    wx:i32[16] <- vk[ww:ww+16]
    wy:i32[16] = add wv wx
    wz:i32[] = add 3072:i32[] vn
    xa:i32[16] <- vk[wz:wz+16]
    xb:i32[16] = add wy xa
    xc:i32[] = add 3328:i32[] vn
    xd:i32[16] <- vk[xc:xc+16]
    xe:i32[16] = add xb xd
    xf:i32[] = add 3584:i32[] vn
    xg:i32[16] <- vk[xf:xf+16]
    xh:i32[16] = add xe xg
    xi:i32[] = add 3840:i32[] vn
    xj:i32[16] <- vk[xi:xi+16]
    xk:i32[16] = add xh xj
    xl:i32[16], vl[vn:vn+16] <- vl[vn:vn+16], xk
  in () } in
let jaxpr8 = { lambda xm:Ref<vmem>{i32[10256]} xn:i32[16] xo:i32[] xp:Ref<vmem>{i32[10256]}
    xq:i32[] xr:i32[] xs:i32[] xt:Ref<vmem>{i32[2064]}; xu:i32[] xv:i32[] xw:i32[]. let
    xx:i32[] = mul xu 16:i32[]
    xy:i32[16] <- xm[xx:xx+16]
    xz:i32[] = mul xu 16:i32[]
    ya:i32[16] = add xz xn
    yb:bool[16] = lt ya xo
    yc:i32[] = mul xu 16:i32[]
    yd:i32[16] <- xp[yc:yc+16]
    ye:i32[16] = shift_right_arithmetic xy xq
    yf:bool[16] = gt ye xr
    yg:bool[16] = and yb yf
    yh:bool[16] = eq ye xr
    yi:bool[16] = and yb yh
    yj:i32[16] = convert_element_type[new_dtype=int32 weak_type=False] yi
    yk:bool[16] = broadcast_in_dim True:bool[]
    yl:i32[16] = masked_cumsum yj yk
    ym:i32[16] = add yl xw
    yn:bool[16] = le ym xs
    yo:bool[16] = and yi yn
    yp:bool[16] = or yg yo
    yq:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] xt yd xv yp
    yr:i32[16] = all_reduce_population_count[reduce=1] yp
    ys:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] yr
    yt:i32[] = squeeze[dimensions=(0,)] ys
    yu:i32[] = add xv yt
    yv:i32[1] = slice[limit_indices=(16,) start_indices=(15,) strides=None] ym
    yw:i32[] = squeeze[dimensions=(0,)] yv
  in (yu, yw) } in
let jaxpr9 = { lambda yx:Ref<vmem>{i32[10240]} yy:i32[16] yz:i32[] za:Ref<vmem>{i32[4112]}; zb:i32[]. let
    zc:i32[] = mul zb 16:i32[]
    zd:i32[16] <- yx[zc:zc+16]
    ze:i32[] = mul zb 16:i32[]
    zf:i32[16] = add ze yy
    zg:bool[16] = lt zf yz
    zh:bool[16] = gt zd -2139095041:i32[]
    zi:bool[16] = and zg zh
    zj:i32[16] = shift_right_arithmetic zd 24:i32[]
    zk:i32[16] = add zj 128:i32[]
    zl:i32[16] = mul yy 256:i32[]
    zm:i32[16] = add zl zk
    zn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] za zm zn zi
  in () } in
let jaxpr10 = { lambda zo:Ref<vmem>{i32[4112]} zp:Ref<vmem>{i32[272]}; zq:i32[]. let
    zr:i32[] = mul zq 16:i32[]
    zs:i32[16] = broadcast_in_dim 0:i32[]
    zt:i32[] = add 0:i32[] zr
    zu:i32[16] <- zo[zt:zt+16]
    zv:i32[16] = add zs zu
    zw:i32[] = add 256:i32[] zr
    zx:i32[16] <- zo[zw:zw+16]
    zy:i32[16] = add zv zx
    zz:i32[] = add 512:i32[] zr
    baa:i32[16] <- zo[zz:zz+16]
    bab:i32[16] = add zy baa
    bac:i32[] = add 768:i32[] zr
    bad:i32[16] <- zo[bac:bac+16]
    bae:i32[16] = add bab bad
    baf:i32[] = add 1024:i32[] zr
    bag:i32[16] <- zo[baf:baf+16]
    bah:i32[16] = add bae bag
    bai:i32[] = add 1280:i32[] zr
    baj:i32[16] <- zo[bai:bai+16]
    bak:i32[16] = add bah baj
    bal:i32[] = add 1536:i32[] zr
    bam:i32[16] <- zo[bal:bal+16]
    ban:i32[16] = add bak bam
    bao:i32[] = add 1792:i32[] zr
    bap:i32[16] <- zo[bao:bao+16]
    baq:i32[16] = add ban bap
    bar:i32[] = add 2048:i32[] zr
    bas:i32[16] <- zo[bar:bar+16]
    bat:i32[16] = add baq bas
    bau:i32[] = add 2304:i32[] zr
    bav:i32[16] <- zo[bau:bau+16]
    baw:i32[16] = add bat bav
    bax:i32[] = add 2560:i32[] zr
    bay:i32[16] <- zo[bax:bax+16]
    baz:i32[16] = add baw bay
    bba:i32[] = add 2816:i32[] zr
    bbb:i32[16] <- zo[bba:bba+16]
    bbc:i32[16] = add baz bbb
    bbd:i32[] = add 3072:i32[] zr
    bbe:i32[16] <- zo[bbd:bbd+16]
    bbf:i32[16] = add bbc bbe
    bbg:i32[] = add 3328:i32[] zr
    bbh:i32[16] <- zo[bbg:bbg+16]
    bbi:i32[16] = add bbf bbh
    bbj:i32[] = add 3584:i32[] zr
    bbk:i32[16] <- zo[bbj:bbj+16]
    bbl:i32[16] = add bbi bbk
    bbm:i32[] = add 3840:i32[] zr
    bbn:i32[16] <- zo[bbm:bbm+16]
    bbo:i32[16] = add bbl bbn
    bbp:i32[16], zp[zr:zr+16] <- zp[zr:zr+16], bbo
  in () } in
let _where4 = { lambda ; bbq:bool[] bbr:bool[16] bbs:bool[16]. let
    bbt:bool[16] = select_n bbq bbs bbr
  in (bbt,) } in
{ lambda ; bbu:Ref<hbm>{i32[10485760]} bbv:Ref<hbm>{i32[1024]} bbw:Ref<hbm>{i32[2097152]}
    bbx:Ref<vmem>{i32[10240]} bby:Ref<vmem>{i32[4112]} bbz:Ref<vmem>{i32[272]} bca:Ref<vmem>{i32[2064]}
    bcb:Ref<vmem>{i32[10256]} bcc:Ref<vmem>{i32[10256]} bcd:Ref<vmem>{i32[1024]}
    bce:Ref<semaphore_mem>{dma_sem[4]}. let
    bcf:i32[] = axis_index[axis_name=core] 
    bcg:i32[] = axis_index[axis_name=subcore] 
    bch:i32[] = mul bcf 16:i32[]
    bci:i32[] = add bch bcg
    bcj:i32[16] = iota[dimension=0 dtype=int32 shape=(16,) sharding=None] 
    dma_start(p0) bbv[:] -> bcd[...] bce[0]
    dma_wait bcd[...] bce[0]
    _:i32[] _:i32[] = scan[
      jaxpr=jaxpr
      length=32
      num_carry=2
      num_consts=12
      reverse=False
      unroll=1
    ] bci bcd bbu bbx bce bby bcj bbz bca bcb bcc bbw 0:i32[] 0:i32[]
  in () }

The kernel jaxpr idx=0 for mpmd_map sc_topk_direct_b1024_n10240_k2048_v3fused at /mnt/pd/home-mp/vllm-torchtpu-test3/vllm-torchtpu/kernel_work/sparsecore-topk/variants/v03-fused-emit/sparsecore_topk.py:447:
let jaxpr = { lambda ; a:i32[] b:Ref<vmem>{i32[1024]} c:Ref<hbm>{i32[10485760]} d:Ref<vmem>{i32[10240]}
    e:Ref<semaphore_mem>{dma_sem[4]} f:Ref<vmem>{i32[4112]} g:i32[16] h:Ref<vmem>{i32[272]}
    i:Ref<vmem>{i32[2064]} j:Ref<vmem>{i32[10256]} k:Ref<vmem>{i32[10256]} l:Ref<hbm>{i32[2097152]}
    m:i32[] n:i32[]. let
    o:i32[] = add m 1:i32[]
    p:i32[] = mul m 32:i32[]
    q:i32[] = convert_element_type[new_dtype=int32 weak_type=False] p
    r:i32[] = add q a
    s:bool[] = lt r 1024:i32[]
    t:i32[] = min r 1023:i32[]
    u:i32[16] <- b[t:t+16]
    v:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] u
    w:i32[] = squeeze[dimensions=(0,)] v
    x:i32[] = min w 10240:i32[]
    y:i32[] = jit[name=_where jaxpr=_where] s x 0:i32[]
    z:i32[] = mul t 10240:i32[]
    ba:bool[] = gt y 0:i32[]
    bb:i32[] = min 10240:i32[] y
    bc:i32[] = jit[name=_where jaxpr=_where] ba bb 0:i32[]
    bd:i32[] = add bc 15:i32[]
    be:i32[] = shift_right_arithmetic bd 4:i32[]
    bf:bool[] = gt bc 0:i32[]
    bg:i32[] = convert_element_type[new_dtype=int32 weak_type=False] bf
    cond[
      branches=(
        { lambda ; bh:Ref<hbm>{i32[10485760]} bi:i32[] bj:Ref<vmem>{i32[10240]} bk:Ref<semaphore_mem>{dma_sem[4]}
            bl:i32[]. let
            
          in () }
        { lambda ; bm:Ref<hbm>{i32[10485760]} bn:i32[] bo:Ref<vmem>{i32[10240]} bp:Ref<semaphore_mem>{dma_sem[4]}
            bq:i32[]. let
            dma_start(p0) bm[bn:bn+10240] -> bo[...] bp[1]
            dma_wait bo[...] bp[1]
            parallel_loop[
              jaxpr={ lambda br:Ref<vmem>{i32[10240]}; bs:i32[]. let
                  bt:i32[] = mul bs 16:i32[]
                  bu:i32[16] <- br[bt:bt+16]
                  bv:i32[16] = shift_right_arithmetic bu 31:i32[]
                  bw:i32[16] = and bv 2147483647:i32[]
                  bx:i32[16] = xor bu bw
                  by:i32[] = mul bs 16:i32[]
                  bz:i32[16], br[by:by+16] <- br[by:by+16], bx
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] bq 1:i32[] bo
          in () }
      )
    ] bg c z d e be
    ca:bool[] = gt bc 0:i32[]
    cb:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ca
    cond[
      branches=(
        { lambda ; cc:Ref<vmem>{i32[4112]} cd:i32[] ce:Ref<vmem>{i32[10240]} cf:i32[16]
            cg:i32[] ch:Ref<vmem>{i32[272]}. let
            
          in () }
        { lambda ; ci:Ref<vmem>{i32[4112]} cj:i32[] ck:Ref<vmem>{i32[10240]} cl:i32[16]
            cm:i32[] cn:Ref<vmem>{i32[272]}. let
            parallel_loop[
              jaxpr={ lambda co:Ref<vmem>{i32[4112]}; cp:i32[]. let
                  cq:i32[16] = broadcast_in_dim 0:i32[]
                  cr:i32[] = mul cp 16:i32[]
                  cs:i32[16], co[cr:cr+16] <- co[cr:cr+16], cq
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] 256:i32[] 1:i32[] ci
            parallel_loop[
              jaxpr=jaxpr9
              tree=PyTreeDef((*, *, *, [*, *, *, *], []))
              unroll=4
            ] 0:i32[] cj 1:i32[] ck cl cm ci
            parallel_loop[
              jaxpr=jaxpr10
              tree=PyTreeDef((*, *, *, [*, *], []))
              unroll=2
            ] 0:i32[] 16:i32[] 1:i32[] ci cn
          in () }
      )
    ] cb f be d g bc h
    ct:bool[] = eq bc 0:i32[]
    cu:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ct
    cond[
      branches=(
        { lambda ; cv:Ref<vmem>{i32[272]}. let  in () }
        { lambda ; cw:Ref<vmem>{i32[272]}. let
            parallel_loop[
              jaxpr={ lambda cx:Ref<vmem>{i32[272]}; cy:i32[]. let
                  cz:i32[16] = broadcast_in_dim 0:i32[]
                  da:i32[] = mul cy 16:i32[]
                  db:i32[16], cx[da:da+16] <- cx[da:da+16], cz
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] 16:i32[] 1:i32[] cw
          in () }
      )
    ] cu h
    dc:i32[] = parallel_loop[
      jaxpr={ lambda dd:Ref<vmem>{i32[272]}; de:i32[] df:i32[]. let
          dg:i32[] = mul de 16:i32[]
          dh:i32[16] <- dd[dg:dg+16]
          di:bool[16] = broadcast_in_dim True:bool[]
          dj:i32[16] = masked_cumsum dh di
          dk:i32[16] = add dj df
          dl:i32[] = mul de 16:i32[]
          dm:i32[16], dd[dl:dl+16] <- dd[dl:dl+16], dk
          dn:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] dk
          do:i32[] = squeeze[dimensions=(0,)] dn
        in (do,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] 16:i32[] 1:i32[] h 0:i32[]
    dp:bool[] = le dc 2048:i32[]
    dq:i32[] = sub dc 2048:i32[]
    dr:i32[] = parallel_loop[
      jaxpr={ lambda ds:Ref<vmem>{i32[272]} dt:i32[]; du:i32[] dv:i32[]. let
          dw:i32[] = mul du 16:i32[]
          dx:i32[16] <- ds[dw:dw+16]
          dy:bool[16] = le dx dt
          dz:i32[16] = all_reduce_population_count[reduce=1] dy
          ea:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] dz
          eb:i32[] = squeeze[dimensions=(0,)] ea
          ec:i32[] = add dv eb
        in (ec,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] 16:i32[] 1:i32[] h dq 0:i32[]
    ed:i32[16] <- h[dr:dr+16]
    ee:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ed
    ef:i32[] = squeeze[dimensions=(0,)] ee
    eg:bool[] = gt dr 0:i32[]
    eh:i32[] = sub dr 1:i32[]
    ei:i32[] = max eh 0:i32[]
    ej:i32[16] <- h[ei:ei+16]
    ek:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ej
    el:i32[] = squeeze[dimensions=(0,)] ek
    em:i32[] = jit[name=_where jaxpr=_where] eg el 0:i32[]
    en:i32[] = sub dc ef
    eo:i32[] = sub ef em
    ep:i32[] = sub 2048:i32[] en
    eq:i32[] = sub dr 128:i32[]
    er:bool[] = eq ep eo
    es:bool[] = or dp er
    et:bool[] = lt dc 2048:i32[]
    eu:bool[] = and s et
    ev:i32[] = jit[name=_where jaxpr=_where1] eu 128:i32[] 0:i32[]
    parallel_loop[
      jaxpr={ lambda ew:Ref<vmem>{i32[2064]}; ex:i32[]. let
          ey:i32[16] = broadcast_in_dim -1:i32[]
          ez:i32[] = mul ex 16:i32[]
          fa:i32[16], ew[ez:ez+16] <- ew[ez:ez+16], ey
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] ev 1:i32[] i
    fb:i32[] = jit[name=_where jaxpr=_where] s be 0:i32[]
    fc:i32[] fd:i32[] = parallel_loop[
      jaxpr=jaxpr1
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] fb 1:i32[] d g bc eq es dp i j k 0:i32[] 0:i32[]
    fe:i32[] = add eo 15:i32[]
    ff:i32[] = shift_right_arithmetic fe 4:i32[]
    fg:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda fh:Ref<vmem>{i32[4112]}; fi:i32[]. let
          fj:i32[16] = broadcast_in_dim 0:i32[]
          fk:i32[] = mul fi 16:i32[]
          fl:i32[16], fh[fk:fk+16] <- fh[fk:fk+16], fj
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] fg 1:i32[] f
    fm:i32[] = jit[name=_where jaxpr=_where2] es 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr2
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] fm 1:i32[] j g eo 24:i32[] eq f
    fn:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr3 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      fn 1:i32[] f h
    fo:i32[] = parallel_loop[
      jaxpr={ lambda fp:Ref<vmem>{i32[272]}; fq:i32[] fr:i32[]. let
          fs:i32[] = mul fq 16:i32[]
          ft:i32[16] <- fp[fs:fs+16]
          fu:bool[16] = broadcast_in_dim True:bool[]
          fv:i32[16] = masked_cumsum ft fu
          fw:i32[16] = add fv fr
          fx:i32[] = mul fq 16:i32[]
          fy:i32[16], fp[fx:fx+16] <- fp[fx:fx+16], fw
          fz:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] fw
          ga:i32[] = squeeze[dimensions=(0,)] fz
        in (ga,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] fn 1:i32[] h 0:i32[]
    gb:i32[] = sub fo ep
    gc:i32[] = parallel_loop[
      jaxpr={ lambda gd:Ref<vmem>{i32[272]} ge:i32[]; gf:i32[] gg:i32[]. let
          gh:i32[] = mul gf 16:i32[]
          gi:i32[16] <- gd[gh:gh+16]
          gj:bool[16] = le gi ge
          gk:i32[16] = all_reduce_population_count[reduce=1] gj
          gl:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gk
          gm:i32[] = squeeze[dimensions=(0,)] gl
          gn:i32[] = add gg gm
        in (gn,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] fn 1:i32[] h gb 0:i32[]
    go:i32[16] <- h[gc:gc+16]
    gp:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] go
    gq:i32[] = squeeze[dimensions=(0,)] gp
    gr:bool[] = gt gc 0:i32[]
    gs:i32[] = sub gc 1:i32[]
    gt:i32[] = max gs 0:i32[]
    gu:i32[16] <- h[gt:gt+16]
    gv:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gu
    gw:i32[] = squeeze[dimensions=(0,)] gv
    gx:i32[] = jit[name=_where jaxpr=_where] gr gw 0:i32[]
    gy:i32[] = sub fo gq
    gz:i32[] = sub gq gx
    ha:i32[] = sub ep gy
    hb:i32[] = sub 24:i32[] 16:i32[]
    hc:i32[] = shift_left eq hb
    hd:i32[] = or hc gc
    he:bool[] = eq ha gz
    hf:bool[] = or es he
    hg:bool[] = eq ha 0:i32[]
    hh:bool[] = or hf hg
    hi:i32[] = jit[name=_where jaxpr=_where3] es eq hd
    hj:i32[] = jit[name=_where jaxpr=_where3] es 24:i32[] 16:i32[]
    hk:i32[] = add en gy
    hl:i32[] = jit[name=_where jaxpr=_where3] es en hk
    hm:i32[] = jit[name=_where jaxpr=_where3] es ep ha
    hn:i32[] = jit[name=_where jaxpr=_where3] es eo gz
    ho:i32[] = jit[name=_where jaxpr=_where1] hh 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda hp:Ref<vmem>{i32[4112]}; hq:i32[]. let
          hr:i32[16] = broadcast_in_dim 0:i32[]
          hs:i32[] = mul hq 16:i32[]
          ht:i32[16], hp[hs:hs+16] <- hp[hs:hs+16], hr
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] ho 1:i32[] f
    hu:i32[] = jit[name=_where jaxpr=_where2] hh 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr4
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] hu 1:i32[] j g hn hj hi f
    hv:i32[] = jit[name=_where jaxpr=_where1] hh 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr5 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      hv 1:i32[] f h
    hw:i32[] = parallel_loop[
      jaxpr={ lambda hx:Ref<vmem>{i32[272]}; hy:i32[] hz:i32[]. let
          ia:i32[] = mul hy 16:i32[]
          ib:i32[16] <- hx[ia:ia+16]
          ic:bool[16] = broadcast_in_dim True:bool[]
          id:i32[16] = masked_cumsum ib ic
          ie:i32[16] = add id hz
          if:i32[] = mul hy 16:i32[]
          ig:i32[16], hx[if:if+16] <- hx[if:if+16], ie
          ih:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] ie
          ii:i32[] = squeeze[dimensions=(0,)] ih
        in (ii,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] hv 1:i32[] h 0:i32[]
    ij:i32[] = sub hw hm
    ik:i32[] = parallel_loop[
      jaxpr={ lambda il:Ref<vmem>{i32[272]} im:i32[]; in:i32[] io:i32[]. let
          ip:i32[] = mul in 16:i32[]
          iq:i32[16] <- il[ip:ip+16]
          ir:bool[16] = le iq im
          is:i32[16] = all_reduce_population_count[reduce=1] ir
          it:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] is
          iu:i32[] = squeeze[dimensions=(0,)] it
          iv:i32[] = add io iu
        in (iv,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] hv 1:i32[] h ij 0:i32[]
    iw:i32[16] <- h[ik:ik+16]
    ix:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] iw
    iy:i32[] = squeeze[dimensions=(0,)] ix
    iz:bool[] = gt ik 0:i32[]
    ja:i32[] = sub ik 1:i32[]
    jb:i32[] = max ja 0:i32[]
    jc:i32[16] <- h[jb:jb+16]
    jd:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] jc
    je:i32[] = squeeze[dimensions=(0,)] jd
    jf:i32[] = jit[name=_where jaxpr=_where] iz je 0:i32[]
    jg:i32[] = sub hw iy
    jh:i32[] = sub iy jf
    ji:i32[] = sub hm jg
    jj:i32[] = sub hj 8:i32[]
    jk:i32[] = shift_left hi jj
    jl:i32[] = or jk ik
    jm:bool[] = eq ji jh
    jn:bool[] = or hh jm
    jo:bool[] = eq ji 0:i32[]
    jp:bool[] = or jn jo
    jq:i32[] = jit[name=_where jaxpr=_where3] hh hi jl
    jr:i32[] = jit[name=_where jaxpr=_where3] hh hj 8:i32[]
    js:i32[] = add hl jg
    jt:i32[] = jit[name=_where jaxpr=_where3] hh hl js
    ju:i32[] = jit[name=_where jaxpr=_where3] hh hm ji
    jv:i32[] = jit[name=_where jaxpr=_where3] hh hn jh
    jw:i32[] = jit[name=_where jaxpr=_where1] jp 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda jx:Ref<vmem>{i32[4112]}; jy:i32[]. let
          jz:i32[16] = broadcast_in_dim 0:i32[]
          ka:i32[] = mul jy 16:i32[]
          kb:i32[16], jx[ka:ka+16] <- jx[ka:ka+16], jz
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] jw 1:i32[] f
    kc:i32[] = jit[name=_where jaxpr=_where2] jp 0:i32[] ff
    parallel_loop[
      jaxpr=jaxpr6
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] kc 1:i32[] j g jv jr jq f
    kd:i32[] = jit[name=_where jaxpr=_where1] jp 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr7 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      kd 1:i32[] f h
    ke:i32[] = parallel_loop[
      jaxpr={ lambda kf:Ref<vmem>{i32[272]}; kg:i32[] kh:i32[]. let
          ki:i32[] = mul kg 16:i32[]
          kj:i32[16] <- kf[ki:ki+16]
          kk:bool[16] = broadcast_in_dim True:bool[]
          kl:i32[16] = masked_cumsum kj kk
          km:i32[16] = add kl kh
          kn:i32[] = mul kg 16:i32[]
          ko:i32[16], kf[kn:kn+16] <- kf[kn:kn+16], km
          kp:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] km
          kq:i32[] = squeeze[dimensions=(0,)] kp
        in (kq,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] kd 1:i32[] h 0:i32[]
    kr:i32[] = sub ke ju
    ks:i32[] = parallel_loop[
      jaxpr={ lambda kt:Ref<vmem>{i32[272]} ku:i32[]; kv:i32[] kw:i32[]. let
          kx:i32[] = mul kv 16:i32[]
          ky:i32[16] <- kt[kx:kx+16]
          kz:bool[16] = le ky ku
          la:i32[16] = all_reduce_population_count[reduce=1] kz
          lb:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] la
          lc:i32[] = squeeze[dimensions=(0,)] lb
          ld:i32[] = add kw lc
        in (ld,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] kd 1:i32[] h kr 0:i32[]
    le:i32[16] <- h[ks:ks+16]
    lf:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] le
    lg:i32[] = squeeze[dimensions=(0,)] lf
    lh:bool[] = gt ks 0:i32[]
    li:i32[] = sub ks 1:i32[]
    lj:i32[] = max li 0:i32[]
    lk:i32[16] <- h[lj:lj+16]
    ll:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] lk
    lm:i32[] = squeeze[dimensions=(0,)] ll
    ln:i32[] = jit[name=_where jaxpr=_where] lh lm 0:i32[]
    lo:i32[] = sub ke lg
    lp:i32[] = sub lg ln
    lq:i32[] = sub ju lo
    lr:i32[] = sub jr 0:i32[]
    ls:i32[] = shift_left jq lr
    lt:i32[] = or ls ks
    lu:bool[] = eq lq lp
    lv:bool[] = or jp lu
    lw:bool[] = eq lq 0:i32[]
    lx:bool[] = or lv lw
    ly:i32[] = jit[name=_where jaxpr=_where3] jp jq lt
    lz:i32[] = jit[name=_where jaxpr=_where3] jp jr 0:i32[]
    ma:i32[] = add jt lo
    mb:i32[] = jit[name=_where jaxpr=_where3] jp jt ma
    mc:i32[] = jit[name=_where jaxpr=_where3] jp ju lq
    md:i32[] = jit[name=_where jaxpr=_where3] jp jv lp
    me:i32[] = jit[name=_where jaxpr=_where2] dp 0:i32[] mc
    mf:bool[] = not es
    mg:bool[] = and s mf
    mh:i32[] = jit[name=_where jaxpr=_where] mg ff 0:i32[]
    mi:i32[] mj:i32[] = parallel_loop[
      jaxpr=jaxpr8
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] mh 1:i32[] j g md k lz ly me i fc 0:i32[]
    mk:i32[] = convert_element_type[new_dtype=int32 weak_type=False] s
    cond[
      branches=(
        { lambda ; ml:i32[] mm:Ref<vmem>{i32[2064]} mn:Ref<hbm>{i32[2097152]} mo:Ref<semaphore_mem>{dma_sem[4]}. let
            
          in () }
        { lambda ; mp:i32[] mq:Ref<vmem>{i32[2064]} mr:Ref<hbm>{i32[2097152]} ms:Ref<semaphore_mem>{dma_sem[4]}. let
            mt:i32[] = mul mp 2048:i32[]
            dma_start(p0) mq[:2048] -> mr[mt:mt+2048] ms[2]
            dma_wait mr[mt:mt+2048] ms[2]
          in () }
      )
    ] mk t i l e
  in (o, 0:i32[]) } in
let _where = { lambda ; mu:bool[] mv:i32[] mw:i32[]. let
    mx:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mw
    my:i32[] = select_n mu mx mv
  in (my,) } in
let _where1 = { lambda ; mz:bool[] na:i32[] nb:i32[]. let
    nc:i32[] = select_n mz nb na
  in (nc,) } in
let jaxpr1 = { lambda nd:Ref<vmem>{i32[10240]} ne:i32[16] nf:i32[] ng:i32[] nh:bool[]
    ni:bool[] nj:Ref<vmem>{i32[2064]} nk:Ref<vmem>{i32[10256]} nl:Ref<vmem>{i32[10256]}; nm:i32[]
    nn:i32[] no:i32[]. let
    np:i32[] = mul nm 16:i32[]
    nq:i32[16] <- nd[np:np+16]
    nr:i32[] = mul nm 16:i32[]
    ns:i32[16] = add nr ne
    nt:bool[16] = lt ns nf
    nu:bool[16] = gt nq -2139095041:i32[]
    nv:bool[16] = and nt nu
    nw:i32[16] = shift_right_arithmetic nq 24:i32[]
    nx:bool[16] = gt nw ng
    ny:bool[16] = and nv nx
    nz:bool[16] = eq nw ng
    oa:bool[16] = and nv nz
    ob:bool[16] = or ny oa
    oc:bool[16] = jit[name=_where jaxpr=_where4] nh ob ny
    od:bool[16] = jit[name=_where jaxpr=_where4] ni nv oc
    oe:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] nj ns nn od
    of:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (10256,), ())], [*]),))
    ] nk nq no oa
    og:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (10256,), ())], [*]),))
    ] nl ns no oa
    oh:i32[16] = all_reduce_population_count[reduce=1] od
    oi:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] oh
    oj:i32[] = squeeze[dimensions=(0,)] oi
    ok:i32[] = add nn oj
    ol:i32[16] = all_reduce_population_count[reduce=1] oa
    om:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ol
    on:i32[] = squeeze[dimensions=(0,)] om
    oo:i32[] = add no on
  in (ok, oo) } in
let _where2 = { lambda ; op:bool[] oq:i32[] or:i32[]. let
    os:i32[] = convert_element_type[new_dtype=int32 weak_type=False] oq
    ot:i32[] = select_n op or os
  in (ot,) } in
let jaxpr2 = { lambda ou:Ref<vmem>{i32[10256]} ov:i32[16] ow:i32[] ox:i32[] oy:i32[]
    oz:Ref<vmem>{i32[4112]}; pa:i32[]. let
    pb:i32[] = mul pa 16:i32[]
    pc:i32[16] <- ou[pb:pb+16]
    pd:i32[] = mul pa 16:i32[]
    pe:i32[16] = add pd ov
    pf:bool[16] = lt pe ow
    pg:i32[16] = shift_right_arithmetic pc ox
    ph:bool[16] = eq pg oy
    pi:bool[16] = and pf ph
    pj:i32[16] = shift_right_arithmetic pc 16:i32[]
    pk:i32[16] = and pj 255:i32[]
    pl:i32[16] = mul ov 256:i32[]
    pm:i32[16] = add pl pk
    pn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] oz pm pn pi
  in () } in
let jaxpr3 = { lambda po:Ref<vmem>{i32[4112]} pp:Ref<vmem>{i32[272]}; pq:i32[]. let
    pr:i32[] = mul pq 16:i32[]
    ps:i32[16] = broadcast_in_dim 0:i32[]
    pt:i32[] = add 0:i32[] pr
    pu:i32[16] <- po[pt:pt+16]
    pv:i32[16] = add ps pu
    pw:i32[] = add 256:i32[] pr
    px:i32[16] <- po[pw:pw+16]
    py:i32[16] = add pv px
    pz:i32[] = add 512:i32[] pr
    qa:i32[16] <- po[pz:pz+16]
    qb:i32[16] = add py qa
    qc:i32[] = add 768:i32[] pr
    qd:i32[16] <- po[qc:qc+16]
    qe:i32[16] = add qb qd
    qf:i32[] = add 1024:i32[] pr
    qg:i32[16] <- po[qf:qf+16]
    qh:i32[16] = add qe qg
    qi:i32[] = add 1280:i32[] pr
    qj:i32[16] <- po[qi:qi+16]
    qk:i32[16] = add qh qj
    ql:i32[] = add 1536:i32[] pr
    qm:i32[16] <- po[ql:ql+16]
    qn:i32[16] = add qk qm
    qo:i32[] = add 1792:i32[] pr
    qp:i32[16] <- po[qo:qo+16]
    qq:i32[16] = add qn qp
    qr:i32[] = add 2048:i32[] pr
    qs:i32[16] <- po[qr:qr+16]
    qt:i32[16] = add qq qs
    qu:i32[] = add 2304:i32[] pr
    qv:i32[16] <- po[qu:qu+16]
    qw:i32[16] = add qt qv
    qx:i32[] = add 2560:i32[] pr
    qy:i32[16] <- po[qx:qx+16]
    qz:i32[16] = add qw qy
    ra:i32[] = add 2816:i32[] pr
    rb:i32[16] <- po[ra:ra+16]
    rc:i32[16] = add qz rb
    rd:i32[] = add 3072:i32[] pr
    re:i32[16] <- po[rd:rd+16]
    rf:i32[16] = add rc re
    rg:i32[] = add 3328:i32[] pr
    rh:i32[16] <- po[rg:rg+16]
    ri:i32[16] = add rf rh
    rj:i32[] = add 3584:i32[] pr
    rk:i32[16] <- po[rj:rj+16]
    rl:i32[16] = add ri rk
    rm:i32[] = add 3840:i32[] pr
    rn:i32[16] <- po[rm:rm+16]
    ro:i32[16] = add rl rn
    rp:i32[16], pp[pr:pr+16] <- pp[pr:pr+16], ro
  in () } in
let _where3 = { lambda ; rq:bool[] rr:i32[] rs:i32[]. let
    rt:i32[] = select_n rq rs rr
  in (rt,) } in
let jaxpr4 = { lambda ru:Ref<vmem>{i32[10256]} rv:i32[16] rw:i32[] rx:i32[] ry:i32[]
    rz:Ref<vmem>{i32[4112]}; sa:i32[]. let
    sb:i32[] = mul sa 16:i32[]
    sc:i32[16] <- ru[sb:sb+16]
    sd:i32[] = mul sa 16:i32[]
    se:i32[16] = add sd rv
    sf:bool[16] = lt se rw
    sg:i32[16] = shift_right_arithmetic sc rx
    sh:bool[16] = eq sg ry
    si:bool[16] = and sf sh
    sj:i32[16] = shift_right_arithmetic sc 8:i32[]
    sk:i32[16] = and sj 255:i32[]
    sl:i32[16] = mul rv 256:i32[]
    sm:i32[16] = add sl sk
    sn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] rz sm sn si
  in () } in
let jaxpr5 = { lambda so:Ref<vmem>{i32[4112]} sp:Ref<vmem>{i32[272]}; sq:i32[]. let
    sr:i32[] = mul sq 16:i32[]
    ss:i32[16] = broadcast_in_dim 0:i32[]
    st:i32[] = add 0:i32[] sr
    su:i32[16] <- so[st:st+16]
    sv:i32[16] = add ss su
    sw:i32[] = add 256:i32[] sr
    sx:i32[16] <- so[sw:sw+16]
    sy:i32[16] = add sv sx
    sz:i32[] = add 512:i32[] sr
    ta:i32[16] <- so[sz:sz+16]
    tb:i32[16] = add sy ta
    tc:i32[] = add 768:i32[] sr
    td:i32[16] <- so[tc:tc+16]
    te:i32[16] = add tb td
    tf:i32[] = add 1024:i32[] sr
    tg:i32[16] <- so[tf:tf+16]
    th:i32[16] = add te tg
    ti:i32[] = add 1280:i32[] sr
    tj:i32[16] <- so[ti:ti+16]
    tk:i32[16] = add th tj
    tl:i32[] = add 1536:i32[] sr
    tm:i32[16] <- so[tl:tl+16]
    tn:i32[16] = add tk tm
    to:i32[] = add 1792:i32[] sr
    tp:i32[16] <- so[to:to+16]
    tq:i32[16] = add tn tp
    tr:i32[] = add 2048:i32[] sr
    ts:i32[16] <- so[tr:tr+16]
    tt:i32[16] = add tq ts
    tu:i32[] = add 2304:i32[] sr
    tv:i32[16] <- so[tu:tu+16]
    tw:i32[16] = add tt tv
    tx:i32[] = add 2560:i32[] sr
    ty:i32[16] <- so[tx:tx+16]
    tz:i32[16] = add tw ty
    ua:i32[] = add 2816:i32[] sr
    ub:i32[16] <- so[ua:ua+16]
    uc:i32[16] = add tz ub
    ud:i32[] = add 3072:i32[] sr
    ue:i32[16] <- so[ud:ud+16]
    uf:i32[16] = add uc ue
    ug:i32[] = add 3328:i32[] sr
    uh:i32[16] <- so[ug:ug+16]
    ui:i32[16] = add uf uh
    uj:i32[] = add 3584:i32[] sr
    uk:i32[16] <- so[uj:uj+16]
    ul:i32[16] = add ui uk
    um:i32[] = add 3840:i32[] sr
    un:i32[16] <- so[um:um+16]
    uo:i32[16] = add ul un
    up:i32[16], sp[sr:sr+16] <- sp[sr:sr+16], uo
  in () } in
let jaxpr6 = { lambda uq:Ref<vmem>{i32[10256]} ur:i32[16] us:i32[] ut:i32[] uu:i32[]
    uv:Ref<vmem>{i32[4112]}; uw:i32[]. let
    ux:i32[] = mul uw 16:i32[]
    uy:i32[16] <- uq[ux:ux+16]
    uz:i32[] = mul uw 16:i32[]
    va:i32[16] = add uz ur
    vb:bool[16] = lt va us
    vc:i32[16] = shift_right_arithmetic uy ut
    vd:bool[16] = eq vc uu
    ve:bool[16] = and vb vd
    vf:i32[16] = shift_right_arithmetic uy 0:i32[]
    vg:i32[16] = and vf 255:i32[]
    vh:i32[16] = mul ur 256:i32[]
    vi:i32[16] = add vh vg
    vj:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] uv vi vj ve
  in () } in
let jaxpr7 = { lambda vk:Ref<vmem>{i32[4112]} vl:Ref<vmem>{i32[272]}; vm:i32[]. let
    vn:i32[] = mul vm 16:i32[]
    vo:i32[16] = broadcast_in_dim 0:i32[]
    vp:i32[] = add 0:i32[] vn
    vq:i32[16] <- vk[vp:vp+16]
    vr:i32[16] = add vo vq
    vs:i32[] = add 256:i32[] vn
    vt:i32[16] <- vk[vs:vs+16]
    vu:i32[16] = add vr vt
    vv:i32[] = add 512:i32[] vn
    vw:i32[16] <- vk[vv:vv+16]
    vx:i32[16] = add vu vw
    vy:i32[] = add 768:i32[] vn
    vz:i32[16] <- vk[vy:vy+16]
    wa:i32[16] = add vx vz
    wb:i32[] = add 1024:i32[] vn
    wc:i32[16] <- vk[wb:wb+16]
    wd:i32[16] = add wa wc
    we:i32[] = add 1280:i32[] vn
    wf:i32[16] <- vk[we:we+16]
    wg:i32[16] = add wd wf
    wh:i32[] = add 1536:i32[] vn
    wi:i32[16] <- vk[wh:wh+16]
    wj:i32[16] = add wg wi
    wk:i32[] = add 1792:i32[] vn
    wl:i32[16] <- vk[wk:wk+16]
    wm:i32[16] = add wj wl
    wn:i32[] = add 2048:i32[] vn
    wo:i32[16] <- vk[wn:wn+16]
    wp:i32[16] = add wm wo
    wq:i32[] = add 2304:i32[] vn
    wr:i32[16] <- vk[wq:wq+16]
    ws:i32[16] = add wp wr
    wt:i32[] = add 2560:i32[] vn
    wu:i32[16] <- vk[wt:wt+16]
    wv:i32[16] = add ws wu
    ww:i32[] = add 2816:i32[] vn
    wx:i32[16] <- vk[ww:ww+16]
    wy:i32[16] = add wv wx
    wz:i32[] = add 3072:i32[] vn
    xa:i32[16] <- vk[wz:wz+16]
    xb:i32[16] = add wy xa
    xc:i32[] = add 3328:i32[] vn
    xd:i32[16] <- vk[xc:xc+16]
    xe:i32[16] = add xb xd
    xf:i32[] = add 3584:i32[] vn
    xg:i32[16] <- vk[xf:xf+16]
    xh:i32[16] = add xe xg
    xi:i32[] = add 3840:i32[] vn
    xj:i32[16] <- vk[xi:xi+16]
    xk:i32[16] = add xh xj
    xl:i32[16], vl[vn:vn+16] <- vl[vn:vn+16], xk
  in () } in
let jaxpr8 = { lambda xm:Ref<vmem>{i32[10256]} xn:i32[16] xo:i32[] xp:Ref<vmem>{i32[10256]}
    xq:i32[] xr:i32[] xs:i32[] xt:Ref<vmem>{i32[2064]}; xu:i32[] xv:i32[] xw:i32[]. let
    xx:i32[] = mul xu 16:i32[]
    xy:i32[16] <- xm[xx:xx+16]
    xz:i32[] = mul xu 16:i32[]
    ya:i32[16] = add xz xn
    yb:bool[16] = lt ya xo
    yc:i32[] = mul xu 16:i32[]
    yd:i32[16] <- xp[yc:yc+16]
    ye:i32[16] = shift_right_arithmetic xy xq
    yf:bool[16] = gt ye xr
    yg:bool[16] = and yb yf
    yh:bool[16] = eq ye xr
    yi:bool[16] = and yb yh
    yj:i32[16] = convert_element_type[new_dtype=int32 weak_type=False] yi
    yk:bool[16] = broadcast_in_dim True:bool[]
    yl:i32[16] = masked_cumsum yj yk
    ym:i32[16] = add yl xw
    yn:bool[16] = le ym xs
    yo:bool[16] = and yi yn
    yp:bool[16] = or yg yo
    yq:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] xt yd xv yp
    yr:i32[16] = all_reduce_population_count[reduce=1] yp
    ys:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] yr
    yt:i32[] = squeeze[dimensions=(0,)] ys
    yu:i32[] = add xv yt
    yv:i32[1] = slice[limit_indices=(16,) start_indices=(15,) strides=None] ym
    yw:i32[] = squeeze[dimensions=(0,)] yv
  in (yu, yw) } in
let jaxpr9 = { lambda yx:Ref<vmem>{i32[10240]} yy:i32[16] yz:i32[] za:Ref<vmem>{i32[4112]}; zb:i32[]. let
    zc:i32[] = mul zb 16:i32[]
    zd:i32[16] <- yx[zc:zc+16]
    ze:i32[] = mul zb 16:i32[]
    zf:i32[16] = add ze yy
    zg:bool[16] = lt zf yz
    zh:bool[16] = gt zd -2139095041:i32[]
    zi:bool[16] = and zg zh
    zj:i32[16] = shift_right_arithmetic zd 24:i32[]
    zk:i32[16] = add zj 128:i32[]
    zl:i32[16] = mul yy 256:i32[]
    zm:i32[16] = add zl zk
    zn:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] za zm zn zi
  in () } in
let jaxpr10 = { lambda zo:Ref<vmem>{i32[4112]} zp:Ref<vmem>{i32[272]}; zq:i32[]. let
    zr:i32[] = mul zq 16:i32[]
    zs:i32[16] = broadcast_in_dim 0:i32[]
    zt:i32[] = add 0:i32[] zr
    zu:i32[16] <- zo[zt:zt+16]
    zv:i32[16] = add zs zu
    zw:i32[] = add 256:i32[] zr
    zx:i32[16] <- zo[zw:zw+16]
    zy:i32[16] = add zv zx
    zz:i32[] = add 512:i32[] zr
    baa:i32[16] <- zo[zz:zz+16]
    bab:i32[16] = add zy baa
    bac:i32[] = add 768:i32[] zr
    bad:i32[16] <- zo[bac:bac+16]
    bae:i32[16] = add bab bad
    baf:i32[] = add 1024:i32[] zr
    bag:i32[16] <- zo[baf:baf+16]
    bah:i32[16] = add bae bag
    bai:i32[] = add 1280:i32[] zr
    baj:i32[16] <- zo[bai:bai+16]
    bak:i32[16] = add bah baj
    bal:i32[] = add 1536:i32[] zr
    bam:i32[16] <- zo[bal:bal+16]
    ban:i32[16] = add bak bam
    bao:i32[] = add 1792:i32[] zr
    bap:i32[16] <- zo[bao:bao+16]
    baq:i32[16] = add ban bap
    bar:i32[] = add 2048:i32[] zr
    bas:i32[16] <- zo[bar:bar+16]
    bat:i32[16] = add baq bas
    bau:i32[] = add 2304:i32[] zr
    bav:i32[16] <- zo[bau:bau+16]
    baw:i32[16] = add bat bav
    bax:i32[] = add 2560:i32[] zr
    bay:i32[16] <- zo[bax:bax+16]
    baz:i32[16] = add baw bay
    bba:i32[] = add 2816:i32[] zr
    bbb:i32[16] <- zo[bba:bba+16]
    bbc:i32[16] = add baz bbb
    bbd:i32[] = add 3072:i32[] zr
    bbe:i32[16] <- zo[bbd:bbd+16]
    bbf:i32[16] = add bbc bbe
    bbg:i32[] = add 3328:i32[] zr
    bbh:i32[16] <- zo[bbg:bbg+16]
    bbi:i32[16] = add bbf bbh
    bbj:i32[] = add 3584:i32[] zr
    bbk:i32[16] <- zo[bbj:bbj+16]
    bbl:i32[16] = add bbi bbk
    bbm:i32[] = add 3840:i32[] zr
    bbn:i32[16] <- zo[bbm:bbm+16]
    bbo:i32[16] = add bbl bbn
    bbp:i32[16], zp[zr:zr+16] <- zp[zr:zr+16], bbo
  in () } in
let _where4 = { lambda ; bbq:bool[] bbr:bool[16] bbs:bool[16]. let
    bbt:bool[16] = select_n bbq bbs bbr
  in (bbt,) } in
{ lambda ; bbu:Ref<hbm>{i32[10485760]} bbv:Ref<hbm>{i32[1024]} bbw:Ref<hbm>{i32[2097152]}
    bbx:Ref<vmem>{i32[10240]} bby:Ref<vmem>{i32[4112]} bbz:Ref<vmem>{i32[272]} bca:Ref<vmem>{i32[2064]}
    bcb:Ref<vmem>{i32[10256]} bcc:Ref<vmem>{i32[10256]} bcd:Ref<vmem>{i32[1024]}
    bce:Ref<semaphore_mem>{dma_sem[4]}. let
    bcf:i32[] = axis_index[axis_name=core] 
    bcg:i32[] = axis_index[axis_name=subcore] 
    bch:i32[] = mul bcf 16:i32[]
    bci:i32[] = add bch bcg
    bcj:i32[16] = iota[dimension=0 dtype=int32 shape=(16,) sharding=None] 
    dma_start(p0) bbv[:] -> bcd[...] bce[0]
    dma_wait bcd[...] bce[0]
    _:i32[] _:i32[] = scan[
      jaxpr=jaxpr
      length=32
      num_carry=2
      num_consts=12
      reverse=False
      unroll=1
    ] bci bcd bbu bbx bce bby bcj bbz bca bcb bcc bbw 0:i32[] 0:i32[]
  in () }

The Mosaic module for mpmd_map:
#map = affine_map<(d0, d1) -> (0)>
module {
  func.func @sc_topk_direct_b1024_n10240_k2048_v3fused(%arg0: i32, %arg1: i32, %arg2: memref<10485760xi32, #tpu.memory_space<hbm>>, %arg3: memref<1024xi32, #tpu.memory_space<hbm>>, %arg4: memref<2097152xi32, #tpu.memory_space<hbm>>, %arg5: memref<10240xi32, #tpu.memory_space<vmem>>, %arg6: memref<4112xi32, #tpu.memory_space<vmem>>, %arg7: memref<272xi32, #tpu.memory_space<vmem>>, %arg8: memref<2064xi32, #tpu.memory_space<vmem>>, %arg9: memref<10256xi32, #tpu.memory_space<vmem>>, %arg10: memref<10256xi32, #tpu.memory_space<vmem>>, %arg11: memref<1024xi32, #tpu.memory_space<vmem>>, %arg12: memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<core_parallel>, #tpu.dimension_semantics<subcore_parallel>], iteration_bounds = array<i64: 2, 16>, scalar_prefetch = 0 : i64, scratch_operands = 8 : i64, tpu.core_type = #tpu.core_type<sc_vector_subcore>, window_params = [{transform_indices = #map}, {transform_indices = #map}, {transform_indices = #map}]} {
    %cst = arith.constant dense<8> : vector<16xi32>
    %cst_0 = arith.constant dense<255> : vector<16xi32>
    %cst_1 = arith.constant dense<16> : vector<16xi32>
    %cst_2 = arith.constant dense<-1> : vector<16xi32>
    %cst_3 = arith.constant dense<true> : vector<16xi1>
    %cst_4 = arith.constant dense<1> : vector<16xi32>
    %cst_5 = arith.constant dense<256> : vector<16xi32>
    %cst_6 = arith.constant dense<128> : vector<16xi32>
    %cst_7 = arith.constant dense<24> : vector<16xi32>
    %cst_8 = arith.constant dense<-2139095041> : vector<16xi32>
    %cst_9 = arith.constant dense<0> : vector<16xi32>
    %cst_10 = arith.constant dense<2147483647> : vector<16xi32>
    %cst_11 = arith.constant dense<31> : vector<16xi32>
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c3840_i32 = arith.constant 3840 : i32
    %c3584_i32 = arith.constant 3584 : i32
    %c3328_i32 = arith.constant 3328 : i32
    %c3072_i32 = arith.constant 3072 : i32
    %c2816_i32 = arith.constant 2816 : i32
    %c2560_i32 = arith.constant 2560 : i32
    %c2304_i32 = arith.constant 2304 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1792_i32 = arith.constant 1792 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c768_i32 = arith.constant 768 : i32
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %c24_i32 = arith.constant 24 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %c15_i32 = arith.constant 15 : i32
    %c10240_i32 = arith.constant 10240 : i32
    %c1023_i32 = arith.constant 1023 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = arith.muli %arg0, %c16_i32 : i32
    %1 = arith.addi %0, %arg1 : i32
    %2 = tpu.iota {dimensions = array<i32: 0>} : vector<16xi32>
    %3 = tpu.memref_slice %arg12[%c0_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %4 = tpu.memref_squeeze %3 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    tpu.enqueue_dma source(%arg3 : memref<1024xi32, #tpu.memory_space<hbm>>) target(%arg11 : memref<1024xi32, #tpu.memory_space<vmem>>) target_semaphore(%4 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    %5 = tpu.memref_slice %arg12[%c0_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %6 = tpu.memref_squeeze %5 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    tpu.wait_dma2 semaphore(%6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%arg3 : memref<1024xi32, #tpu.memory_space<hbm>>) dst(%arg11 : memref<1024xi32, #tpu.memory_space<vmem>>)
    scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %7 = arith.muli %arg13, %c32_i32 : i32
      %8 = arith.addi %7, %1 : i32
      %9 = arith.cmpi slt, %8, %c1024_i32 : i32
      %10 = arith.minsi %8, %c1023_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = tpu.vector_load %arg11[%11] {strides = array<i32>} : memref<1024xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %13 = vector.extract %12[0] : i32 from vector<16xi32>
      %14 = arith.minsi %13, %c10240_i32 : i32
      %15 = arith.select %9, %14, %c0_i32 : i32
      %16 = arith.muli %10, %c10240_i32 : i32
      %17 = arith.cmpi sgt, %15, %c0_i32 : i32
      %18 = arith.minsi %15, %c10240_i32 : i32
      %19 = arith.select %17, %18, %c0_i32 : i32
      %20 = arith.addi %19, %c15_i32 : i32
      %21 = arith.shrsi %20, %c4_i32 : i32
      %22 = arith.cmpi sgt, %19, %c0_i32 : i32
      scf.if %22 {
        %141 = tpu.memref_slice %arg12[%c1_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %142 = tpu.memref_squeeze %141 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %143 = tpu.memref_slice %arg2[%16] : memref<10485760xi32, #tpu.memory_space<hbm>> -> memref<10240xi32, #tpu.memory_space<hbm>>
        tpu.enqueue_dma source(%143 : memref<10240xi32, #tpu.memory_space<hbm>>) target(%arg5 : memref<10240xi32, #tpu.memory_space<vmem>>) target_semaphore(%142 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
        %144 = tpu.memref_slice %arg12[%c1_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %145 = tpu.memref_squeeze %144 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %146 = tpu.memref_slice %arg2[%16] : memref<10485760xi32, #tpu.memory_space<hbm>> -> memref<10240xi32, #tpu.memory_space<hbm>>
        tpu.wait_dma2 semaphore(%145 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%146 : memref<10240xi32, #tpu.memory_space<hbm>>) dst(%arg5 : memref<10240xi32, #tpu.memory_space<vmem>>)
        scf.for %arg14 = %c0_i32 to %21 step %c1_i32  : i32 {
          %147 = arith.muli %arg14, %c16_i32 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = tpu.vector_load %arg5[%148] {strides = array<i32>} : memref<10240xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %150 = arith.shrsi %149, %cst_11 : vector<16xi32>
          %151 = arith.andi %150, %cst_10 : vector<16xi32>
          %152 = arith.xori %149, %151 : vector<16xi32>
          %153 = arith.muli %arg14, %c16_i32 : i32
          %154 = arith.index_cast %153 : i32 to index
          tpu.vector_store %arg5[%154], %152 {strides = array<i32>} : memref<10240xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      }
      %23 = arith.cmpi sgt, %19, %c0_i32 : i32
      scf.if %23 {
        scf.for %arg14 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
          %141 = arith.muli %arg14, %c16_i32 : i32
          %142 = arith.index_cast %141 : i32 to index
          tpu.vector_store %arg6[%142], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
        scf.for %arg14 = %c0_i32 to %21 step %c1_i32  : i32 {
          %141 = arith.muli %arg14, %c16_i32 : i32
          %142 = arith.index_cast %141 : i32 to index
          %143 = tpu.vector_load %arg5[%142] {strides = array<i32>} : memref<10240xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %144 = arith.muli %arg14, %c16_i32 : i32
          %145 = vector.broadcast %144 : i32 to vector<16xi32>
          %146 = arith.addi %145, %2 : vector<16xi32>
          %147 = vector.broadcast %19 : i32 to vector<16xi32>
          %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
          %149 = arith.cmpi sgt, %143, %cst_8 : vector<16xi32>
          %150 = arith.andi %148, %149 : vector<16xi1>
          %151 = arith.shrsi %143, %cst_7 : vector<16xi32>
          %152 = arith.addi %151, %cst_6 : vector<16xi32>
          %153 = arith.muli %2, %cst_5 : vector<16xi32>
          %154 = arith.addi %153, %152 : vector<16xi32>
          tpu.vector_store_idx %arg6[%154], %cst_4 masked %150 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
        scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %141 = arith.muli %arg14, %c16_i32 : i32
          %142 = arith.index_cast %141 : i32 to index
          %143 = tpu.vector_load %arg6[%142] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %144 = arith.addi %141, %c256_i32 : i32
          %145 = arith.index_cast %144 : i32 to index
          %146 = tpu.vector_load %arg6[%145] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %147 = arith.addi %143, %146 : vector<16xi32>
          %148 = arith.addi %141, %c512_i32 : i32
          %149 = arith.index_cast %148 : i32 to index
          %150 = tpu.vector_load %arg6[%149] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %151 = arith.addi %147, %150 : vector<16xi32>
          %152 = arith.addi %141, %c768_i32 : i32
          %153 = arith.index_cast %152 : i32 to index
          %154 = tpu.vector_load %arg6[%153] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %155 = arith.addi %151, %154 : vector<16xi32>
          %156 = arith.addi %141, %c1024_i32 : i32
          %157 = arith.index_cast %156 : i32 to index
          %158 = tpu.vector_load %arg6[%157] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %159 = arith.addi %155, %158 : vector<16xi32>
          %160 = arith.addi %141, %c1280_i32 : i32
          %161 = arith.index_cast %160 : i32 to index
          %162 = tpu.vector_load %arg6[%161] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %163 = arith.addi %159, %162 : vector<16xi32>
          %164 = arith.addi %141, %c1536_i32 : i32
          %165 = arith.index_cast %164 : i32 to index
          %166 = tpu.vector_load %arg6[%165] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %167 = arith.addi %163, %166 : vector<16xi32>
          %168 = arith.addi %141, %c1792_i32 : i32
          %169 = arith.index_cast %168 : i32 to index
          %170 = tpu.vector_load %arg6[%169] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %171 = arith.addi %167, %170 : vector<16xi32>
          %172 = arith.addi %141, %c2048_i32 : i32
          %173 = arith.index_cast %172 : i32 to index
          %174 = tpu.vector_load %arg6[%173] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %175 = arith.addi %171, %174 : vector<16xi32>
          %176 = arith.addi %141, %c2304_i32 : i32
          %177 = arith.index_cast %176 : i32 to index
          %178 = tpu.vector_load %arg6[%177] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %179 = arith.addi %175, %178 : vector<16xi32>
          %180 = arith.addi %141, %c2560_i32 : i32
          %181 = arith.index_cast %180 : i32 to index
          %182 = tpu.vector_load %arg6[%181] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %183 = arith.addi %179, %182 : vector<16xi32>
          %184 = arith.addi %141, %c2816_i32 : i32
          %185 = arith.index_cast %184 : i32 to index
          %186 = tpu.vector_load %arg6[%185] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %187 = arith.addi %183, %186 : vector<16xi32>
          %188 = arith.addi %141, %c3072_i32 : i32
          %189 = arith.index_cast %188 : i32 to index
          %190 = tpu.vector_load %arg6[%189] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %191 = arith.addi %187, %190 : vector<16xi32>
          %192 = arith.addi %141, %c3328_i32 : i32
          %193 = arith.index_cast %192 : i32 to index
          %194 = tpu.vector_load %arg6[%193] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %195 = arith.addi %191, %194 : vector<16xi32>
          %196 = arith.addi %141, %c3584_i32 : i32
          %197 = arith.index_cast %196 : i32 to index
          %198 = tpu.vector_load %arg6[%197] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %199 = arith.addi %195, %198 : vector<16xi32>
          %200 = arith.addi %141, %c3840_i32 : i32
          %201 = arith.index_cast %200 : i32 to index
          %202 = tpu.vector_load %arg6[%201] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %203 = arith.addi %199, %202 : vector<16xi32>
          %204 = arith.index_cast %141 : i32 to index
          tpu.vector_store %arg7[%204], %203 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      }
      %24 = arith.cmpi eq, %19, %c0_i32 : i32
      scf.if %24 {
        scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %141 = arith.muli %arg14, %c16_i32 : i32
          %142 = arith.index_cast %141 : i32 to index
          tpu.vector_store %arg7[%142], %cst_9 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      }
      %25 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = tpu.scan <sum>, %143 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %145 = vector.broadcast %arg15 : i32 to vector<16xi32>
        %146 = arith.addi %144, %145 : vector<16xi32>
        %147 = arith.muli %arg14, %c16_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        tpu.vector_store %arg7[%148], %146 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %149 = vector.extract %146[15] : i32 from vector<16xi32>
        scf.yield %149 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %26 = arith.cmpi sle, %25, %c2048_i32 : i32
      %27 = arith.subi %25, %c2048_i32 : i32
      %28 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = vector.broadcast %27 : i32 to vector<16xi32>
        %145 = arith.cmpi sle, %143, %144 : vector<16xi32>
        %146 = tpu.all_reduce %145 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %147 = vector.extract %146[0] : i32 from vector<16xi32>
        %148 = arith.addi %arg15, %147 : i32
        scf.yield %148 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %29 = arith.index_cast %28 : i32 to index
      %30 = tpu.vector_load %arg7[%29] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %31 = vector.extract %30[0] : i32 from vector<16xi32>
      %32 = arith.cmpi sgt, %28, %c0_i32 : i32
      %33 = arith.subi %28, %c1_i32 : i32
      %34 = arith.maxsi %33, %c0_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = tpu.vector_load %arg7[%35] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %37 = vector.extract %36[0] : i32 from vector<16xi32>
      %38 = arith.select %32, %37, %c0_i32 : i32
      %39 = arith.subi %25, %31 : i32
      %40 = arith.subi %31, %38 : i32
      %41 = arith.subi %c2048_i32, %39 : i32
      %42 = arith.subi %28, %c128_i32 : i32
      %43 = arith.cmpi eq, %41, %40 : i32
      %44 = arith.ori %26, %43 : i1
      %45 = arith.cmpi slt, %25, %c2048_i32 : i32
      %46 = arith.andi %9, %45 : i1
      %47 = arith.select %46, %c128_i32, %c0_i32 : i32
      scf.for %arg14 = %c0_i32 to %47 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg8[%142], %cst_2 {strides = array<i32>} : memref<2064xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %48 = arith.select %9, %21, %c0_i32 : i32
      %49:2 = scf.for %arg14 = %c0_i32 to %48 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg5[%142] {strides = array<i32>} : memref<10240xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.muli %arg14, %c16_i32 : i32
        %145 = vector.broadcast %144 : i32 to vector<16xi32>
        %146 = arith.addi %145, %2 : vector<16xi32>
        %147 = vector.broadcast %19 : i32 to vector<16xi32>
        %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
        %149 = arith.cmpi sgt, %143, %cst_8 : vector<16xi32>
        %150 = arith.andi %148, %149 : vector<16xi1>
        %151 = arith.shrsi %143, %cst_7 : vector<16xi32>
        %152 = vector.broadcast %42 : i32 to vector<16xi32>
        %153 = arith.cmpi sgt, %151, %152 : vector<16xi32>
        %154 = arith.andi %150, %153 : vector<16xi1>
        %155 = vector.broadcast %42 : i32 to vector<16xi32>
        %156 = arith.cmpi eq, %151, %155 : vector<16xi32>
        %157 = arith.andi %150, %156 : vector<16xi1>
        %158 = arith.ori %154, %157 : vector<16xi1>
        %159 = arith.select %44, %158, %154 : vector<16xi1>
        %160 = arith.select %26, %150, %159 : vector<16xi1>
        %161 = arith.index_cast %arg15 : i32 to index
        tpu.vector_store %arg8[%161], %146 masked %160 {strides = array<i32>} : memref<2064xi32, #tpu.memory_space<vmem>>, vector<16xi32>, vector<16xi1>
        %162 = arith.index_cast %arg16 : i32 to index
        tpu.vector_store %arg9[%162], %143 masked %157 {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, vector<16xi1>
        %163 = arith.index_cast %arg16 : i32 to index
        tpu.vector_store %arg10[%163], %146 masked %157 {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, vector<16xi1>
        %164 = tpu.all_reduce %160 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %165 = vector.extract %164[0] : i32 from vector<16xi32>
        %166 = arith.addi %arg15, %165 : i32
        %167 = tpu.all_reduce %157 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %168 = vector.extract %167[0] : i32 from vector<16xi32>
        %169 = arith.addi %arg16, %168 : i32
        scf.yield %166, %169 : i32, i32
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %50 = arith.addi %40, %c15_i32 : i32
      %51 = arith.shrsi %50, %c4_i32 : i32
      %52 = arith.select %44, %c0_i32, %c256_i32 : i32
      scf.for %arg14 = %c0_i32 to %52 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg6[%142], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %53 = arith.select %44, %c0_i32, %51 : i32
      scf.for %arg14 = %c0_i32 to %53 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg9[%142] {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.muli %arg14, %c16_i32 : i32
        %145 = vector.broadcast %144 : i32 to vector<16xi32>
        %146 = arith.addi %145, %2 : vector<16xi32>
        %147 = vector.broadcast %40 : i32 to vector<16xi32>
        %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
        %149 = arith.shrsi %143, %cst_7 : vector<16xi32>
        %150 = vector.broadcast %42 : i32 to vector<16xi32>
        %151 = arith.cmpi eq, %149, %150 : vector<16xi32>
        %152 = arith.andi %148, %151 : vector<16xi1>
        %153 = arith.shrsi %143, %cst_1 : vector<16xi32>
        %154 = arith.andi %153, %cst_0 : vector<16xi32>
        %155 = arith.muli %2, %cst_5 : vector<16xi32>
        %156 = arith.addi %155, %154 : vector<16xi32>
        tpu.vector_store_idx %arg6[%156], %cst_4 masked %152 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %54 = arith.select %44, %c0_i32, %c16_i32 : i32
      scf.for %arg14 = %c0_i32 to %54 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg6[%142] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.addi %141, %c256_i32 : i32
        %145 = arith.index_cast %144 : i32 to index
        %146 = tpu.vector_load %arg6[%145] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %147 = arith.addi %143, %146 : vector<16xi32>
        %148 = arith.addi %141, %c512_i32 : i32
        %149 = arith.index_cast %148 : i32 to index
        %150 = tpu.vector_load %arg6[%149] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %151 = arith.addi %147, %150 : vector<16xi32>
        %152 = arith.addi %141, %c768_i32 : i32
        %153 = arith.index_cast %152 : i32 to index
        %154 = tpu.vector_load %arg6[%153] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %155 = arith.addi %151, %154 : vector<16xi32>
        %156 = arith.addi %141, %c1024_i32 : i32
        %157 = arith.index_cast %156 : i32 to index
        %158 = tpu.vector_load %arg6[%157] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %159 = arith.addi %155, %158 : vector<16xi32>
        %160 = arith.addi %141, %c1280_i32 : i32
        %161 = arith.index_cast %160 : i32 to index
        %162 = tpu.vector_load %arg6[%161] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %163 = arith.addi %159, %162 : vector<16xi32>
        %164 = arith.addi %141, %c1536_i32 : i32
        %165 = arith.index_cast %164 : i32 to index
        %166 = tpu.vector_load %arg6[%165] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %167 = arith.addi %163, %166 : vector<16xi32>
        %168 = arith.addi %141, %c1792_i32 : i32
        %169 = arith.index_cast %168 : i32 to index
        %170 = tpu.vector_load %arg6[%169] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %171 = arith.addi %167, %170 : vector<16xi32>
        %172 = arith.addi %141, %c2048_i32 : i32
        %173 = arith.index_cast %172 : i32 to index
        %174 = tpu.vector_load %arg6[%173] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %175 = arith.addi %171, %174 : vector<16xi32>
        %176 = arith.addi %141, %c2304_i32 : i32
        %177 = arith.index_cast %176 : i32 to index
        %178 = tpu.vector_load %arg6[%177] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %179 = arith.addi %175, %178 : vector<16xi32>
        %180 = arith.addi %141, %c2560_i32 : i32
        %181 = arith.index_cast %180 : i32 to index
        %182 = tpu.vector_load %arg6[%181] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %183 = arith.addi %179, %182 : vector<16xi32>
        %184 = arith.addi %141, %c2816_i32 : i32
        %185 = arith.index_cast %184 : i32 to index
        %186 = tpu.vector_load %arg6[%185] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %187 = arith.addi %183, %186 : vector<16xi32>
        %188 = arith.addi %141, %c3072_i32 : i32
        %189 = arith.index_cast %188 : i32 to index
        %190 = tpu.vector_load %arg6[%189] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %191 = arith.addi %187, %190 : vector<16xi32>
        %192 = arith.addi %141, %c3328_i32 : i32
        %193 = arith.index_cast %192 : i32 to index
        %194 = tpu.vector_load %arg6[%193] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %195 = arith.addi %191, %194 : vector<16xi32>
        %196 = arith.addi %141, %c3584_i32 : i32
        %197 = arith.index_cast %196 : i32 to index
        %198 = tpu.vector_load %arg6[%197] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %199 = arith.addi %195, %198 : vector<16xi32>
        %200 = arith.addi %141, %c3840_i32 : i32
        %201 = arith.index_cast %200 : i32 to index
        %202 = tpu.vector_load %arg6[%201] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %203 = arith.addi %199, %202 : vector<16xi32>
        %204 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg7[%204], %203 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %55 = scf.for %arg14 = %c0_i32 to %54 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = tpu.scan <sum>, %143 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %145 = vector.broadcast %arg15 : i32 to vector<16xi32>
        %146 = arith.addi %144, %145 : vector<16xi32>
        %147 = arith.muli %arg14, %c16_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        tpu.vector_store %arg7[%148], %146 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %149 = vector.extract %146[15] : i32 from vector<16xi32>
        scf.yield %149 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %56 = arith.subi %55, %41 : i32
      %57 = scf.for %arg14 = %c0_i32 to %54 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = vector.broadcast %56 : i32 to vector<16xi32>
        %145 = arith.cmpi sle, %143, %144 : vector<16xi32>
        %146 = tpu.all_reduce %145 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %147 = vector.extract %146[0] : i32 from vector<16xi32>
        %148 = arith.addi %arg15, %147 : i32
        scf.yield %148 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %58 = arith.index_cast %57 : i32 to index
      %59 = tpu.vector_load %arg7[%58] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %60 = vector.extract %59[0] : i32 from vector<16xi32>
      %61 = arith.cmpi sgt, %57, %c0_i32 : i32
      %62 = arith.subi %57, %c1_i32 : i32
      %63 = arith.maxsi %62, %c0_i32 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = tpu.vector_load %arg7[%64] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %66 = vector.extract %65[0] : i32 from vector<16xi32>
      %67 = arith.select %61, %66, %c0_i32 : i32
      %68 = arith.subi %55, %60 : i32
      %69 = arith.subi %60, %67 : i32
      %70 = arith.subi %41, %68 : i32
      %71 = arith.shli %42, %c8_i32 : i32
      %72 = arith.ori %71, %57 : i32
      %73 = arith.cmpi eq, %70, %69 : i32
      %74 = arith.ori %44, %73 : i1
      %75 = arith.cmpi eq, %70, %c0_i32 : i32
      %76 = arith.ori %74, %75 : i1
      %77 = arith.select %44, %42, %72 : i32
      %78 = arith.select %44, %c24_i32, %c16_i32 : i32
      %79 = arith.select %44, %41, %70 : i32
      %80 = arith.select %44, %40, %69 : i32
      %81 = arith.select %76, %c0_i32, %c256_i32 : i32
      scf.for %arg14 = %c0_i32 to %81 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg6[%142], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %82 = arith.select %76, %c0_i32, %51 : i32
      scf.for %arg14 = %c0_i32 to %82 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg9[%142] {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.muli %arg14, %c16_i32 : i32
        %145 = vector.broadcast %144 : i32 to vector<16xi32>
        %146 = arith.addi %145, %2 : vector<16xi32>
        %147 = vector.broadcast %80 : i32 to vector<16xi32>
        %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
        %149 = vector.broadcast %78 : i32 to vector<16xi32>
        %150 = arith.shrsi %143, %149 : vector<16xi32>
        %151 = vector.broadcast %77 : i32 to vector<16xi32>
        %152 = arith.cmpi eq, %150, %151 : vector<16xi32>
        %153 = arith.andi %148, %152 : vector<16xi1>
        %154 = arith.shrsi %143, %cst : vector<16xi32>
        %155 = arith.andi %154, %cst_0 : vector<16xi32>
        %156 = arith.muli %2, %cst_5 : vector<16xi32>
        %157 = arith.addi %156, %155 : vector<16xi32>
        tpu.vector_store_idx %arg6[%157], %cst_4 masked %153 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %83 = arith.select %76, %c0_i32, %c16_i32 : i32
      scf.for %arg14 = %c0_i32 to %83 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg6[%142] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.addi %141, %c256_i32 : i32
        %145 = arith.index_cast %144 : i32 to index
        %146 = tpu.vector_load %arg6[%145] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %147 = arith.addi %143, %146 : vector<16xi32>
        %148 = arith.addi %141, %c512_i32 : i32
        %149 = arith.index_cast %148 : i32 to index
        %150 = tpu.vector_load %arg6[%149] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %151 = arith.addi %147, %150 : vector<16xi32>
        %152 = arith.addi %141, %c768_i32 : i32
        %153 = arith.index_cast %152 : i32 to index
        %154 = tpu.vector_load %arg6[%153] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %155 = arith.addi %151, %154 : vector<16xi32>
        %156 = arith.addi %141, %c1024_i32 : i32
        %157 = arith.index_cast %156 : i32 to index
        %158 = tpu.vector_load %arg6[%157] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %159 = arith.addi %155, %158 : vector<16xi32>
        %160 = arith.addi %141, %c1280_i32 : i32
        %161 = arith.index_cast %160 : i32 to index
        %162 = tpu.vector_load %arg6[%161] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %163 = arith.addi %159, %162 : vector<16xi32>
        %164 = arith.addi %141, %c1536_i32 : i32
        %165 = arith.index_cast %164 : i32 to index
        %166 = tpu.vector_load %arg6[%165] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %167 = arith.addi %163, %166 : vector<16xi32>
        %168 = arith.addi %141, %c1792_i32 : i32
        %169 = arith.index_cast %168 : i32 to index
        %170 = tpu.vector_load %arg6[%169] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %171 = arith.addi %167, %170 : vector<16xi32>
        %172 = arith.addi %141, %c2048_i32 : i32
        %173 = arith.index_cast %172 : i32 to index
        %174 = tpu.vector_load %arg6[%173] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %175 = arith.addi %171, %174 : vector<16xi32>
        %176 = arith.addi %141, %c2304_i32 : i32
        %177 = arith.index_cast %176 : i32 to index
        %178 = tpu.vector_load %arg6[%177] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %179 = arith.addi %175, %178 : vector<16xi32>
        %180 = arith.addi %141, %c2560_i32 : i32
        %181 = arith.index_cast %180 : i32 to index
        %182 = tpu.vector_load %arg6[%181] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %183 = arith.addi %179, %182 : vector<16xi32>
        %184 = arith.addi %141, %c2816_i32 : i32
        %185 = arith.index_cast %184 : i32 to index
        %186 = tpu.vector_load %arg6[%185] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %187 = arith.addi %183, %186 : vector<16xi32>
        %188 = arith.addi %141, %c3072_i32 : i32
        %189 = arith.index_cast %188 : i32 to index
        %190 = tpu.vector_load %arg6[%189] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %191 = arith.addi %187, %190 : vector<16xi32>
        %192 = arith.addi %141, %c3328_i32 : i32
        %193 = arith.index_cast %192 : i32 to index
        %194 = tpu.vector_load %arg6[%193] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %195 = arith.addi %191, %194 : vector<16xi32>
        %196 = arith.addi %141, %c3584_i32 : i32
        %197 = arith.index_cast %196 : i32 to index
        %198 = tpu.vector_load %arg6[%197] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %199 = arith.addi %195, %198 : vector<16xi32>
        %200 = arith.addi %141, %c3840_i32 : i32
        %201 = arith.index_cast %200 : i32 to index
        %202 = tpu.vector_load %arg6[%201] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %203 = arith.addi %199, %202 : vector<16xi32>
        %204 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg7[%204], %203 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %84 = scf.for %arg14 = %c0_i32 to %83 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = tpu.scan <sum>, %143 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %145 = vector.broadcast %arg15 : i32 to vector<16xi32>
        %146 = arith.addi %144, %145 : vector<16xi32>
        %147 = arith.muli %arg14, %c16_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        tpu.vector_store %arg7[%148], %146 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %149 = vector.extract %146[15] : i32 from vector<16xi32>
        scf.yield %149 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %85 = arith.subi %84, %79 : i32
      %86 = scf.for %arg14 = %c0_i32 to %83 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = vector.broadcast %85 : i32 to vector<16xi32>
        %145 = arith.cmpi sle, %143, %144 : vector<16xi32>
        %146 = tpu.all_reduce %145 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %147 = vector.extract %146[0] : i32 from vector<16xi32>
        %148 = arith.addi %arg15, %147 : i32
        scf.yield %148 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %87 = arith.index_cast %86 : i32 to index
      %88 = tpu.vector_load %arg7[%87] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %89 = vector.extract %88[0] : i32 from vector<16xi32>
      %90 = arith.cmpi sgt, %86, %c0_i32 : i32
      %91 = arith.subi %86, %c1_i32 : i32
      %92 = arith.maxsi %91, %c0_i32 : i32
      %93 = arith.index_cast %92 : i32 to index
      %94 = tpu.vector_load %arg7[%93] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %95 = vector.extract %94[0] : i32 from vector<16xi32>
      %96 = arith.select %90, %95, %c0_i32 : i32
      %97 = arith.subi %84, %89 : i32
      %98 = arith.subi %89, %96 : i32
      %99 = arith.subi %79, %97 : i32
      %100 = arith.subi %78, %c8_i32 : i32
      %101 = arith.shli %77, %100 : i32
      %102 = arith.ori %101, %86 : i32
      %103 = arith.cmpi eq, %99, %98 : i32
      %104 = arith.ori %76, %103 : i1
      %105 = arith.cmpi eq, %99, %c0_i32 : i32
      %106 = arith.ori %104, %105 : i1
      %107 = arith.select %76, %77, %102 : i32
      %108 = arith.select %76, %78, %c8_i32 : i32
      %109 = arith.select %76, %79, %99 : i32
      %110 = arith.select %76, %80, %98 : i32
      %111 = arith.select %106, %c0_i32, %c256_i32 : i32
      scf.for %arg14 = %c0_i32 to %111 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg6[%142], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %112 = arith.select %106, %c0_i32, %51 : i32
      scf.for %arg14 = %c0_i32 to %112 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg9[%142] {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.muli %arg14, %c16_i32 : i32
        %145 = vector.broadcast %144 : i32 to vector<16xi32>
        %146 = arith.addi %145, %2 : vector<16xi32>
        %147 = vector.broadcast %110 : i32 to vector<16xi32>
        %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
        %149 = vector.broadcast %108 : i32 to vector<16xi32>
        %150 = arith.shrsi %143, %149 : vector<16xi32>
        %151 = vector.broadcast %107 : i32 to vector<16xi32>
        %152 = arith.cmpi eq, %150, %151 : vector<16xi32>
        %153 = arith.andi %148, %152 : vector<16xi1>
        %154 = arith.andi %143, %cst_0 : vector<16xi32>
        %155 = arith.muli %2, %cst_5 : vector<16xi32>
        %156 = arith.addi %155, %154 : vector<16xi32>
        tpu.vector_store_idx %arg6[%156], %cst_4 masked %153 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %113 = arith.select %106, %c0_i32, %c16_i32 : i32
      scf.for %arg14 = %c0_i32 to %113 step %c1_i32  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg6[%142] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.addi %141, %c256_i32 : i32
        %145 = arith.index_cast %144 : i32 to index
        %146 = tpu.vector_load %arg6[%145] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %147 = arith.addi %143, %146 : vector<16xi32>
        %148 = arith.addi %141, %c512_i32 : i32
        %149 = arith.index_cast %148 : i32 to index
        %150 = tpu.vector_load %arg6[%149] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %151 = arith.addi %147, %150 : vector<16xi32>
        %152 = arith.addi %141, %c768_i32 : i32
        %153 = arith.index_cast %152 : i32 to index
        %154 = tpu.vector_load %arg6[%153] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %155 = arith.addi %151, %154 : vector<16xi32>
        %156 = arith.addi %141, %c1024_i32 : i32
        %157 = arith.index_cast %156 : i32 to index
        %158 = tpu.vector_load %arg6[%157] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %159 = arith.addi %155, %158 : vector<16xi32>
        %160 = arith.addi %141, %c1280_i32 : i32
        %161 = arith.index_cast %160 : i32 to index
        %162 = tpu.vector_load %arg6[%161] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %163 = arith.addi %159, %162 : vector<16xi32>
        %164 = arith.addi %141, %c1536_i32 : i32
        %165 = arith.index_cast %164 : i32 to index
        %166 = tpu.vector_load %arg6[%165] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %167 = arith.addi %163, %166 : vector<16xi32>
        %168 = arith.addi %141, %c1792_i32 : i32
        %169 = arith.index_cast %168 : i32 to index
        %170 = tpu.vector_load %arg6[%169] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %171 = arith.addi %167, %170 : vector<16xi32>
        %172 = arith.addi %141, %c2048_i32 : i32
        %173 = arith.index_cast %172 : i32 to index
        %174 = tpu.vector_load %arg6[%173] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %175 = arith.addi %171, %174 : vector<16xi32>
        %176 = arith.addi %141, %c2304_i32 : i32
        %177 = arith.index_cast %176 : i32 to index
        %178 = tpu.vector_load %arg6[%177] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %179 = arith.addi %175, %178 : vector<16xi32>
        %180 = arith.addi %141, %c2560_i32 : i32
        %181 = arith.index_cast %180 : i32 to index
        %182 = tpu.vector_load %arg6[%181] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %183 = arith.addi %179, %182 : vector<16xi32>
        %184 = arith.addi %141, %c2816_i32 : i32
        %185 = arith.index_cast %184 : i32 to index
        %186 = tpu.vector_load %arg6[%185] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %187 = arith.addi %183, %186 : vector<16xi32>
        %188 = arith.addi %141, %c3072_i32 : i32
        %189 = arith.index_cast %188 : i32 to index
        %190 = tpu.vector_load %arg6[%189] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %191 = arith.addi %187, %190 : vector<16xi32>
        %192 = arith.addi %141, %c3328_i32 : i32
        %193 = arith.index_cast %192 : i32 to index
        %194 = tpu.vector_load %arg6[%193] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %195 = arith.addi %191, %194 : vector<16xi32>
        %196 = arith.addi %141, %c3584_i32 : i32
        %197 = arith.index_cast %196 : i32 to index
        %198 = tpu.vector_load %arg6[%197] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %199 = arith.addi %195, %198 : vector<16xi32>
        %200 = arith.addi %141, %c3840_i32 : i32
        %201 = arith.index_cast %200 : i32 to index
        %202 = tpu.vector_load %arg6[%201] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %203 = arith.addi %199, %202 : vector<16xi32>
        %204 = arith.index_cast %141 : i32 to index
        tpu.vector_store %arg7[%204], %203 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %114 = scf.for %arg14 = %c0_i32 to %113 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = tpu.scan <sum>, %143 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %145 = vector.broadcast %arg15 : i32 to vector<16xi32>
        %146 = arith.addi %144, %145 : vector<16xi32>
        %147 = arith.muli %arg14, %c16_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        tpu.vector_store %arg7[%148], %146 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %149 = vector.extract %146[15] : i32 from vector<16xi32>
        scf.yield %149 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %115 = arith.subi %114, %109 : i32
      %116 = scf.for %arg14 = %c0_i32 to %113 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg7[%142] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = vector.broadcast %115 : i32 to vector<16xi32>
        %145 = arith.cmpi sle, %143, %144 : vector<16xi32>
        %146 = tpu.all_reduce %145 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %147 = vector.extract %146[0] : i32 from vector<16xi32>
        %148 = arith.addi %arg15, %147 : i32
        scf.yield %148 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %117 = arith.index_cast %116 : i32 to index
      %118 = tpu.vector_load %arg7[%117] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %119 = vector.extract %118[0] : i32 from vector<16xi32>
      %120 = arith.cmpi sgt, %116, %c0_i32 : i32
      %121 = arith.subi %116, %c1_i32 : i32
      %122 = arith.maxsi %121, %c0_i32 : i32
      %123 = arith.index_cast %122 : i32 to index
      %124 = tpu.vector_load %arg7[%123] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %125 = vector.extract %124[0] : i32 from vector<16xi32>
      %126 = arith.select %120, %125, %c0_i32 : i32
      %127 = arith.subi %114, %119 : i32
      %128 = arith.subi %119, %126 : i32
      %129 = arith.subi %109, %127 : i32
      %130 = arith.shli %107, %108 : i32
      %131 = arith.ori %130, %116 : i32
      %132 = arith.select %106, %107, %131 : i32
      %133 = arith.select %106, %108, %c0_i32 : i32
      %134 = arith.select %106, %109, %129 : i32
      %135 = arith.select %106, %110, %128 : i32
      %136 = arith.select %26, %c0_i32, %134 : i32
      %137 = arith.xori %44, %true : i1
      %138 = arith.andi %9, %137 : i1
      %139 = arith.select %138, %51, %c0_i32 : i32
      %140:2 = scf.for %arg14 = %c0_i32 to %139 step %c1_i32 iter_args(%arg15 = %49#0, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
        %141 = arith.muli %arg14, %c16_i32 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tpu.vector_load %arg9[%142] {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %144 = arith.muli %arg14, %c16_i32 : i32
        %145 = vector.broadcast %144 : i32 to vector<16xi32>
        %146 = arith.addi %145, %2 : vector<16xi32>
        %147 = vector.broadcast %135 : i32 to vector<16xi32>
        %148 = arith.cmpi slt, %146, %147 : vector<16xi32>
        %149 = arith.muli %arg14, %c16_i32 : i32
        %150 = arith.index_cast %149 : i32 to index
        %151 = tpu.vector_load %arg10[%150] {strides = array<i32>} : memref<10256xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %152 = vector.broadcast %133 : i32 to vector<16xi32>
        %153 = arith.shrsi %143, %152 : vector<16xi32>
        %154 = vector.broadcast %132 : i32 to vector<16xi32>
        %155 = arith.cmpi sgt, %153, %154 : vector<16xi32>
        %156 = arith.andi %148, %155 : vector<16xi1>
        %157 = vector.broadcast %132 : i32 to vector<16xi32>
        %158 = arith.cmpi eq, %153, %157 : vector<16xi32>
        %159 = arith.andi %148, %158 : vector<16xi1>
        %160 = arith.extui %159 : vector<16xi1> to vector<16xi32>
        %161 = tpu.scan <sum>, %160 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %162 = vector.broadcast %arg16 : i32 to vector<16xi32>
        %163 = arith.addi %161, %162 : vector<16xi32>
        %164 = vector.broadcast %136 : i32 to vector<16xi32>
        %165 = arith.cmpi sle, %163, %164 : vector<16xi32>
        %166 = arith.andi %159, %165 : vector<16xi1>
        %167 = arith.ori %156, %166 : vector<16xi1>
        %168 = arith.index_cast %arg15 : i32 to index
        tpu.vector_store %arg8[%168], %151 masked %167 {strides = array<i32>} : memref<2064xi32, #tpu.memory_space<vmem>>, vector<16xi32>, vector<16xi1>
        %169 = tpu.all_reduce %167 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %170 = vector.extract %169[0] : i32 from vector<16xi32>
        %171 = arith.addi %arg15, %170 : i32
        %172 = vector.extract %163[15] : i32 from vector<16xi32>
        scf.yield %171, %172 : i32, i32
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      scf.if %9 {
        %141 = arith.muli %10, %c2048_i32 : i32
        %142 = tpu.memref_slice %arg12[%c2_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %143 = tpu.memref_squeeze %142 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %144 = tpu.memref_slice %arg4[%141] : memref<2097152xi32, #tpu.memory_space<hbm>> -> memref<2048xi32, #tpu.memory_space<hbm>>
        %145 = tpu.memref_slice %arg8[%c0_i32] : memref<2064xi32, #tpu.memory_space<vmem>> -> memref<2048xi32, #tpu.memory_space<vmem>>
        tpu.enqueue_dma source(%145 : memref<2048xi32, #tpu.memory_space<vmem>>) target(%144 : memref<2048xi32, #tpu.memory_space<hbm>>) target_semaphore(%143 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
        %146 = tpu.memref_slice %arg12[%c2_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %147 = tpu.memref_squeeze %146 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %148 = tpu.memref_slice %arg4[%141] : memref<2097152xi32, #tpu.memory_space<hbm>> -> memref<2048xi32, #tpu.memory_space<hbm>>
        %149 = tpu.memref_slice %arg8[%c0_i32] : memref<2064xi32, #tpu.memory_space<vmem>> -> memref<2048xi32, #tpu.memory_space<vmem>>
        tpu.wait_dma2 semaphore(%147 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%149 : memref<2048xi32, #tpu.memory_space<vmem>>) dst(%148 : memref<2048xi32, #tpu.memory_space<hbm>>)
      }
    }
    return
  }
}

