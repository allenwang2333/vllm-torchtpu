jaxpr for sc_vector_subcore
let jaxpr = { lambda ; a:i32[] b:Ref<vmem>{i32[1024]} c:Ref<hbm>{f32[10485760]} d:Ref<vmem>{f32[10240]}
    e:Ref<semaphore_mem>{dma_sem[4]} f:Ref<vmem>{i32[4112]} g:i32[16] h:Ref<vmem>{i32[272]}
    i:Ref<vmem>{i32[2064]} j:Ref<hbm>{i32[2097152]} k:i32[] l:i32[]. let
    m:i32[] = add k 1:i32[]
    n:i32[] = mul k 32:i32[]
    o:i32[] = convert_element_type[new_dtype=int32 weak_type=False] n
    p:i32[] = add o a
    q:bool[] = lt p 1024:i32[]
    r:i32[] = min p 1023:i32[]
    s:i32[16] <- b[r:r+16]
    t:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] s
    u:i32[] = squeeze[dimensions=(0,)] t
    v:i32[] = min u 10240:i32[]
    w:i32[] = jit[name=_where jaxpr=_where] q v 0:i32[]
    x:i32[] = mul r 10240:i32[]
    y:bool[] = gt w 0:i32[]
    z:i32[] = min 10240:i32[] w
    ba:i32[] = jit[name=_where jaxpr=_where] y z 0:i32[]
    bb:i32[] = add ba 15:i32[]
    bc:i32[] = shift_right_arithmetic bb 4:i32[]
    bd:bool[] = gt ba 0:i32[]
    be:i32[] = convert_element_type[new_dtype=int32 weak_type=False] bd
    cond[
      branches=(
        { lambda ; bf:Ref<hbm>{f32[10485760]} bg:i32[] bh:Ref<vmem>{f32[10240]} bi:Ref<semaphore_mem>{dma_sem[4]}
            bj:i32[]. let
            
          in () }
        { lambda ; bk:Ref<hbm>{f32[10485760]} bl:i32[] bm:Ref<vmem>{f32[10240]} bn:Ref<semaphore_mem>{dma_sem[4]}
            bo:i32[]. let
            dma_start(p0) bk[bl:bl+10240] -> bm[...] bn[1]
            dma_wait bm[...] bn[1]
            parallel_loop[
              jaxpr={ lambda bp:Ref<vmem>{f32[10240]}; bq:i32[]. let
                  br:i32[] = mul bq 16:i32[]
                  bs:f32[16] <- bp[br:br+16]
                  bt:i32[16] = bitcast_convert_type[new_dtype=int32] bs
                  bu:i32[16] = shift_right_arithmetic bt 31:i32[]
                  bv:i32[16] = and bu 2147483647:i32[]
                  bw:i32[16] = xor bt bv
                  bx:f32[16] = bitcast_convert_type[new_dtype=float32] bw
                  by:i32[] = mul bq 16:i32[]
                  bz:f32[16], bp[by:by+16] <- bp[by:by+16], bx
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] bo 1:i32[] bm
          in () }
      )
    ] be c x d e bc
    ca:bool[] = gt ba 0:i32[]
    cb:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ca
    cond[
      branches=(
        { lambda ; cc:Ref<vmem>{i32[4112]} cd:i32[] ce:Ref<vmem>{f32[10240]} cf:i32[16]
            cg:i32[] ch:Ref<vmem>{i32[272]}. let
            
          in () }
        { lambda ; ci:Ref<vmem>{i32[4112]} cj:i32[] ck:Ref<vmem>{f32[10240]} cl:i32[16]
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
              jaxpr=jaxpr8
              tree=PyTreeDef((*, *, *, [*, *, *, *], []))
              unroll=4
            ] 0:i32[] cj 1:i32[] ck cl cm ci
            parallel_loop[
              jaxpr=jaxpr9
              tree=PyTreeDef((*, *, *, [*, *], []))
              unroll=2
            ] 0:i32[] 16:i32[] 1:i32[] ci cn
          in () }
      )
    ] cb f bc d g ba h
    ct:bool[] = eq ba 0:i32[]
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
    et:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda eu:Ref<vmem>{i32[4112]}; ev:i32[]. let
          ew:i32[16] = broadcast_in_dim 0:i32[]
          ex:i32[] = mul ev 16:i32[]
          ey:i32[16], eu[ex:ex+16] <- eu[ex:ex+16], ew
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] et 1:i32[] f
    ez:i32[] = jit[name=_where jaxpr=_where2] es 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr1
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] ez 1:i32[] d g ba 24:i32[] eq f
    fa:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr2 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      fa 1:i32[] f h
    fb:i32[] = parallel_loop[
      jaxpr={ lambda fc:Ref<vmem>{i32[272]}; fd:i32[] fe:i32[]. let
          ff:i32[] = mul fd 16:i32[]
          fg:i32[16] <- fc[ff:ff+16]
          fh:bool[16] = broadcast_in_dim True:bool[]
          fi:i32[16] = masked_cumsum fg fh
          fj:i32[16] = add fi fe
          fk:i32[] = mul fd 16:i32[]
          fl:i32[16], fc[fk:fk+16] <- fc[fk:fk+16], fj
          fm:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] fj
          fn:i32[] = squeeze[dimensions=(0,)] fm
        in (fn,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] fa 1:i32[] h 0:i32[]
    fo:i32[] = sub fb ep
    fp:i32[] = parallel_loop[
      jaxpr={ lambda fq:Ref<vmem>{i32[272]} fr:i32[]; fs:i32[] ft:i32[]. let
          fu:i32[] = mul fs 16:i32[]
          fv:i32[16] <- fq[fu:fu+16]
          fw:bool[16] = le fv fr
          fx:i32[16] = all_reduce_population_count[reduce=1] fw
          fy:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] fx
          fz:i32[] = squeeze[dimensions=(0,)] fy
          ga:i32[] = add ft fz
        in (ga,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] fa 1:i32[] h fo 0:i32[]
    gb:i32[16] <- h[fp:fp+16]
    gc:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gb
    gd:i32[] = squeeze[dimensions=(0,)] gc
    ge:bool[] = gt fp 0:i32[]
    gf:i32[] = sub fp 1:i32[]
    gg:i32[] = max gf 0:i32[]
    gh:i32[16] <- h[gg:gg+16]
    gi:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gh
    gj:i32[] = squeeze[dimensions=(0,)] gi
    gk:i32[] = jit[name=_where jaxpr=_where] ge gj 0:i32[]
    gl:i32[] = sub fb gd
    gm:i32[] = sub gd gk
    gn:i32[] = sub ep gl
    go:i32[] = sub 24:i32[] 16:i32[]
    gp:i32[] = shift_left eq go
    gq:i32[] = or gp fp
    gr:bool[] = eq gn gm
    gs:bool[] = or es gr
    gt:bool[] = eq gn 0:i32[]
    gu:bool[] = or gs gt
    gv:i32[] = jit[name=_where jaxpr=_where3] es eq gq
    gw:i32[] = jit[name=_where jaxpr=_where3] es 24:i32[] 16:i32[]
    gx:i32[] = add en gl
    gy:i32[] = jit[name=_where jaxpr=_where3] es en gx
    gz:i32[] = jit[name=_where jaxpr=_where3] es ep gn
    ha:i32[] = jit[name=_where jaxpr=_where3] es eo gm
    hb:i32[] = jit[name=_where jaxpr=_where1] gu 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda hc:Ref<vmem>{i32[4112]}; hd:i32[]. let
          he:i32[16] = broadcast_in_dim 0:i32[]
          hf:i32[] = mul hd 16:i32[]
          hg:i32[16], hc[hf:hf+16] <- hc[hf:hf+16], he
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] hb 1:i32[] f
    hh:i32[] = jit[name=_where jaxpr=_where2] gu 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr3
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] hh 1:i32[] d g ba gw gv f
    hi:i32[] = jit[name=_where jaxpr=_where1] gu 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr4 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      hi 1:i32[] f h
    hj:i32[] = parallel_loop[
      jaxpr={ lambda hk:Ref<vmem>{i32[272]}; hl:i32[] hm:i32[]. let
          hn:i32[] = mul hl 16:i32[]
          ho:i32[16] <- hk[hn:hn+16]
          hp:bool[16] = broadcast_in_dim True:bool[]
          hq:i32[16] = masked_cumsum ho hp
          hr:i32[16] = add hq hm
          hs:i32[] = mul hl 16:i32[]
          ht:i32[16], hk[hs:hs+16] <- hk[hs:hs+16], hr
          hu:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] hr
          hv:i32[] = squeeze[dimensions=(0,)] hu
        in (hv,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] hi 1:i32[] h 0:i32[]
    hw:i32[] = sub hj gz
    hx:i32[] = parallel_loop[
      jaxpr={ lambda hy:Ref<vmem>{i32[272]} hz:i32[]; ia:i32[] ib:i32[]. let
          ic:i32[] = mul ia 16:i32[]
          id:i32[16] <- hy[ic:ic+16]
          ie:bool[16] = le id hz
          if:i32[16] = all_reduce_population_count[reduce=1] ie
          ig:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] if
          ih:i32[] = squeeze[dimensions=(0,)] ig
          ii:i32[] = add ib ih
        in (ii,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] hi 1:i32[] h hw 0:i32[]
    ij:i32[16] <- h[hx:hx+16]
    ik:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ij
    il:i32[] = squeeze[dimensions=(0,)] ik
    im:bool[] = gt hx 0:i32[]
    in:i32[] = sub hx 1:i32[]
    io:i32[] = max in 0:i32[]
    ip:i32[16] <- h[io:io+16]
    iq:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ip
    ir:i32[] = squeeze[dimensions=(0,)] iq
    is:i32[] = jit[name=_where jaxpr=_where] im ir 0:i32[]
    it:i32[] = sub hj il
    iu:i32[] = sub il is
    iv:i32[] = sub gz it
    iw:i32[] = sub gw 8:i32[]
    ix:i32[] = shift_left gv iw
    iy:i32[] = or ix hx
    iz:bool[] = eq iv iu
    ja:bool[] = or gu iz
    jb:bool[] = eq iv 0:i32[]
    jc:bool[] = or ja jb
    jd:i32[] = jit[name=_where jaxpr=_where3] gu gv iy
    je:i32[] = jit[name=_where jaxpr=_where3] gu gw 8:i32[]
    jf:i32[] = add gy it
    jg:i32[] = jit[name=_where jaxpr=_where3] gu gy jf
    jh:i32[] = jit[name=_where jaxpr=_where3] gu gz iv
    ji:i32[] = jit[name=_where jaxpr=_where3] gu ha iu
    jj:i32[] = jit[name=_where jaxpr=_where1] jc 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda jk:Ref<vmem>{i32[4112]}; jl:i32[]. let
          jm:i32[16] = broadcast_in_dim 0:i32[]
          jn:i32[] = mul jl 16:i32[]
          jo:i32[16], jk[jn:jn+16] <- jk[jn:jn+16], jm
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] jj 1:i32[] f
    jp:i32[] = jit[name=_where jaxpr=_where2] jc 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr5
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] jp 1:i32[] d g ba je jd f
    jq:i32[] = jit[name=_where jaxpr=_where1] jc 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr6 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      jq 1:i32[] f h
    jr:i32[] = parallel_loop[
      jaxpr={ lambda js:Ref<vmem>{i32[272]}; jt:i32[] ju:i32[]. let
          jv:i32[] = mul jt 16:i32[]
          jw:i32[16] <- js[jv:jv+16]
          jx:bool[16] = broadcast_in_dim True:bool[]
          jy:i32[16] = masked_cumsum jw jx
          jz:i32[16] = add jy ju
          ka:i32[] = mul jt 16:i32[]
          kb:i32[16], js[ka:ka+16] <- js[ka:ka+16], jz
          kc:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] jz
          kd:i32[] = squeeze[dimensions=(0,)] kc
        in (kd,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] jq 1:i32[] h 0:i32[]
    ke:i32[] = sub jr jh
    kf:i32[] = parallel_loop[
      jaxpr={ lambda kg:Ref<vmem>{i32[272]} kh:i32[]; ki:i32[] kj:i32[]. let
          kk:i32[] = mul ki 16:i32[]
          kl:i32[16] <- kg[kk:kk+16]
          km:bool[16] = le kl kh
          kn:i32[16] = all_reduce_population_count[reduce=1] km
          ko:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kn
          kp:i32[] = squeeze[dimensions=(0,)] ko
          kq:i32[] = add kj kp
        in (kq,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] jq 1:i32[] h ke 0:i32[]
    kr:i32[16] <- h[kf:kf+16]
    ks:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kr
    kt:i32[] = squeeze[dimensions=(0,)] ks
    ku:bool[] = gt kf 0:i32[]
    kv:i32[] = sub kf 1:i32[]
    kw:i32[] = max kv 0:i32[]
    kx:i32[16] <- h[kw:kw+16]
    ky:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kx
    kz:i32[] = squeeze[dimensions=(0,)] ky
    la:i32[] = jit[name=_where jaxpr=_where] ku kz 0:i32[]
    lb:i32[] = sub jr kt
    lc:i32[] = sub kt la
    ld:i32[] = sub jh lb
    le:i32[] = sub je 0:i32[]
    lf:i32[] = shift_left jd le
    lg:i32[] = or lf kf
    lh:bool[] = eq ld lc
    li:bool[] = or jc lh
    lj:bool[] = eq ld 0:i32[]
    lk:bool[] = or li lj
    ll:i32[] = jit[name=_where jaxpr=_where3] jc jd lg
    lm:i32[] = jit[name=_where jaxpr=_where3] jc je 0:i32[]
    ln:i32[] = add jg lb
    lo:i32[] = jit[name=_where jaxpr=_where3] jc jg ln
    lp:i32[] = jit[name=_where jaxpr=_where3] jc jh ld
    lq:i32[] = jit[name=_where jaxpr=_where3] jc ji lc
    lr:i32[] = jit[name=_where jaxpr=_where2] dp 0:i32[] lp
    ls:bool[] = lt dc 2048:i32[]
    lt:bool[] = and q ls
    lu:i32[] = jit[name=_where jaxpr=_where1] lt 128:i32[] 0:i32[]
    parallel_loop[
      jaxpr={ lambda lv:Ref<vmem>{i32[2064]}; lw:i32[]. let
          lx:i32[16] = broadcast_in_dim -1:i32[]
          ly:i32[] = mul lw 16:i32[]
          lz:i32[16], lv[ly:ly+16] <- lv[ly:ly+16], lx
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] lu 1:i32[] i
    ma:i32[] = jit[name=_where jaxpr=_where] q bc 0:i32[]
    mb:i32[] mc:i32[] = parallel_loop[
      jaxpr=jaxpr7
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] ma 1:i32[] d g ba lm ll lr dp i 0:i32[] 0:i32[]
    md:i32[] = convert_element_type[new_dtype=int32 weak_type=False] q
    cond[
      branches=(
        { lambda ; me:i32[] mf:Ref<vmem>{i32[2064]} mg:Ref<hbm>{i32[2097152]} mh:Ref<semaphore_mem>{dma_sem[4]}. let
            
          in () }
        { lambda ; mi:i32[] mj:Ref<vmem>{i32[2064]} mk:Ref<hbm>{i32[2097152]} ml:Ref<semaphore_mem>{dma_sem[4]}. let
            mm:i32[] = mul mi 2048:i32[]
            dma_start(p0) mj[:2048] -> mk[mm:mm+2048] ml[2]
            dma_wait mk[mm:mm+2048] ml[2]
          in () }
      )
    ] md r i j e
  in (m, 0:i32[]) } in
let _where = { lambda ; mn:bool[] mo:i32[] mp:i32[]. let
    mq:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mp
    mr:i32[] = select_n mn mq mo
  in (mr,) } in
let _where1 = { lambda ; ms:bool[] mt:i32[] mu:i32[]. let
    mv:i32[] = select_n ms mu mt
  in (mv,) } in
let _where2 = { lambda ; mw:bool[] mx:i32[] my:i32[]. let
    mz:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mx
    na:i32[] = select_n mw my mz
  in (na,) } in
let jaxpr1 = { lambda nb:Ref<vmem>{f32[10240]} nc:i32[16] nd:i32[] ne:i32[] nf:i32[]
    ng:Ref<vmem>{i32[4112]}; nh:i32[]. let
    ni:i32[] = mul nh 16:i32[]
    nj:f32[16] <- nb[ni:ni+16]
    nk:i32[16] = bitcast_convert_type[new_dtype=int32] nj
    nl:i32[] = mul nh 16:i32[]
    nm:i32[16] = add nl nc
    nn:bool[16] = lt nm nd
    no:bool[16] = gt nk -2139095041:i32[]
    np:bool[16] = and nn no
    nq:i32[16] = shift_right_arithmetic nk ne
    nr:bool[16] = eq nq nf
    ns:bool[16] = and np nr
    nt:i32[16] = shift_right_arithmetic nk 16:i32[]
    nu:i32[16] = and nt 255:i32[]
    nv:i32[16] = mul nc 256:i32[]
    nw:i32[16] = add nv nu
    nx:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] ng nw nx ns
  in () } in
let jaxpr2 = { lambda ny:Ref<vmem>{i32[4112]} nz:Ref<vmem>{i32[272]}; oa:i32[]. let
    ob:i32[] = mul oa 16:i32[]
    oc:i32[16] = broadcast_in_dim 0:i32[]
    od:i32[] = add 0:i32[] ob
    oe:i32[16] <- ny[od:od+16]
    of:i32[16] = add oc oe
    og:i32[] = add 256:i32[] ob
    oh:i32[16] <- ny[og:og+16]
    oi:i32[16] = add of oh
    oj:i32[] = add 512:i32[] ob
    ok:i32[16] <- ny[oj:oj+16]
    ol:i32[16] = add oi ok
    om:i32[] = add 768:i32[] ob
    on:i32[16] <- ny[om:om+16]
    oo:i32[16] = add ol on
    op:i32[] = add 1024:i32[] ob
    oq:i32[16] <- ny[op:op+16]
    or:i32[16] = add oo oq
    os:i32[] = add 1280:i32[] ob
    ot:i32[16] <- ny[os:os+16]
    ou:i32[16] = add or ot
    ov:i32[] = add 1536:i32[] ob
    ow:i32[16] <- ny[ov:ov+16]
    ox:i32[16] = add ou ow
    oy:i32[] = add 1792:i32[] ob
    oz:i32[16] <- ny[oy:oy+16]
    pa:i32[16] = add ox oz
    pb:i32[] = add 2048:i32[] ob
    pc:i32[16] <- ny[pb:pb+16]
    pd:i32[16] = add pa pc
    pe:i32[] = add 2304:i32[] ob
    pf:i32[16] <- ny[pe:pe+16]
    pg:i32[16] = add pd pf
    ph:i32[] = add 2560:i32[] ob
    pi:i32[16] <- ny[ph:ph+16]
    pj:i32[16] = add pg pi
    pk:i32[] = add 2816:i32[] ob
    pl:i32[16] <- ny[pk:pk+16]
    pm:i32[16] = add pj pl
    pn:i32[] = add 3072:i32[] ob
    po:i32[16] <- ny[pn:pn+16]
    pp:i32[16] = add pm po
    pq:i32[] = add 3328:i32[] ob
    pr:i32[16] <- ny[pq:pq+16]
    ps:i32[16] = add pp pr
    pt:i32[] = add 3584:i32[] ob
    pu:i32[16] <- ny[pt:pt+16]
    pv:i32[16] = add ps pu
    pw:i32[] = add 3840:i32[] ob
    px:i32[16] <- ny[pw:pw+16]
    py:i32[16] = add pv px
    pz:i32[16], nz[ob:ob+16] <- nz[ob:ob+16], py
  in () } in
let _where3 = { lambda ; qa:bool[] qb:i32[] qc:i32[]. let
    qd:i32[] = select_n qa qc qb
  in (qd,) } in
let jaxpr3 = { lambda qe:Ref<vmem>{f32[10240]} qf:i32[16] qg:i32[] qh:i32[] qi:i32[]
    qj:Ref<vmem>{i32[4112]}; qk:i32[]. let
    ql:i32[] = mul qk 16:i32[]
    qm:f32[16] <- qe[ql:ql+16]
    qn:i32[16] = bitcast_convert_type[new_dtype=int32] qm
    qo:i32[] = mul qk 16:i32[]
    qp:i32[16] = add qo qf
    qq:bool[16] = lt qp qg
    qr:bool[16] = gt qn -2139095041:i32[]
    qs:bool[16] = and qq qr
    qt:i32[16] = shift_right_arithmetic qn qh
    qu:bool[16] = eq qt qi
    qv:bool[16] = and qs qu
    qw:i32[16] = shift_right_arithmetic qn 8:i32[]
    qx:i32[16] = and qw 255:i32[]
    qy:i32[16] = mul qf 256:i32[]
    qz:i32[16] = add qy qx
    ra:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] qj qz ra qv
  in () } in
let jaxpr4 = { lambda rb:Ref<vmem>{i32[4112]} rc:Ref<vmem>{i32[272]}; rd:i32[]. let
    re:i32[] = mul rd 16:i32[]
    rf:i32[16] = broadcast_in_dim 0:i32[]
    rg:i32[] = add 0:i32[] re
    rh:i32[16] <- rb[rg:rg+16]
    ri:i32[16] = add rf rh
    rj:i32[] = add 256:i32[] re
    rk:i32[16] <- rb[rj:rj+16]
    rl:i32[16] = add ri rk
    rm:i32[] = add 512:i32[] re
    rn:i32[16] <- rb[rm:rm+16]
    ro:i32[16] = add rl rn
    rp:i32[] = add 768:i32[] re
    rq:i32[16] <- rb[rp:rp+16]
    rr:i32[16] = add ro rq
    rs:i32[] = add 1024:i32[] re
    rt:i32[16] <- rb[rs:rs+16]
    ru:i32[16] = add rr rt
    rv:i32[] = add 1280:i32[] re
    rw:i32[16] <- rb[rv:rv+16]
    rx:i32[16] = add ru rw
    ry:i32[] = add 1536:i32[] re
    rz:i32[16] <- rb[ry:ry+16]
    sa:i32[16] = add rx rz
    sb:i32[] = add 1792:i32[] re
    sc:i32[16] <- rb[sb:sb+16]
    sd:i32[16] = add sa sc
    se:i32[] = add 2048:i32[] re
    sf:i32[16] <- rb[se:se+16]
    sg:i32[16] = add sd sf
    sh:i32[] = add 2304:i32[] re
    si:i32[16] <- rb[sh:sh+16]
    sj:i32[16] = add sg si
    sk:i32[] = add 2560:i32[] re
    sl:i32[16] <- rb[sk:sk+16]
    sm:i32[16] = add sj sl
    sn:i32[] = add 2816:i32[] re
    so:i32[16] <- rb[sn:sn+16]
    sp:i32[16] = add sm so
    sq:i32[] = add 3072:i32[] re
    sr:i32[16] <- rb[sq:sq+16]
    ss:i32[16] = add sp sr
    st:i32[] = add 3328:i32[] re
    su:i32[16] <- rb[st:st+16]
    sv:i32[16] = add ss su
    sw:i32[] = add 3584:i32[] re
    sx:i32[16] <- rb[sw:sw+16]
    sy:i32[16] = add sv sx
    sz:i32[] = add 3840:i32[] re
    ta:i32[16] <- rb[sz:sz+16]
    tb:i32[16] = add sy ta
    tc:i32[16], rc[re:re+16] <- rc[re:re+16], tb
  in () } in
let jaxpr5 = { lambda td:Ref<vmem>{f32[10240]} te:i32[16] tf:i32[] tg:i32[] th:i32[]
    ti:Ref<vmem>{i32[4112]}; tj:i32[]. let
    tk:i32[] = mul tj 16:i32[]
    tl:f32[16] <- td[tk:tk+16]
    tm:i32[16] = bitcast_convert_type[new_dtype=int32] tl
    tn:i32[] = mul tj 16:i32[]
    to:i32[16] = add tn te
    tp:bool[16] = lt to tf
    tq:bool[16] = gt tm -2139095041:i32[]
    tr:bool[16] = and tp tq
    ts:i32[16] = shift_right_arithmetic tm tg
    tt:bool[16] = eq ts th
    tu:bool[16] = and tr tt
    tv:i32[16] = shift_right_arithmetic tm 0:i32[]
    tw:i32[16] = and tv 255:i32[]
    tx:i32[16] = mul te 256:i32[]
    ty:i32[16] = add tx tw
    tz:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] ti ty tz tu
  in () } in
let jaxpr6 = { lambda ua:Ref<vmem>{i32[4112]} ub:Ref<vmem>{i32[272]}; uc:i32[]. let
    ud:i32[] = mul uc 16:i32[]
    ue:i32[16] = broadcast_in_dim 0:i32[]
    uf:i32[] = add 0:i32[] ud
    ug:i32[16] <- ua[uf:uf+16]
    uh:i32[16] = add ue ug
    ui:i32[] = add 256:i32[] ud
    uj:i32[16] <- ua[ui:ui+16]
    uk:i32[16] = add uh uj
    ul:i32[] = add 512:i32[] ud
    um:i32[16] <- ua[ul:ul+16]
    un:i32[16] = add uk um
    uo:i32[] = add 768:i32[] ud
    up:i32[16] <- ua[uo:uo+16]
    uq:i32[16] = add un up
    ur:i32[] = add 1024:i32[] ud
    us:i32[16] <- ua[ur:ur+16]
    ut:i32[16] = add uq us
    uu:i32[] = add 1280:i32[] ud
    uv:i32[16] <- ua[uu:uu+16]
    uw:i32[16] = add ut uv
    ux:i32[] = add 1536:i32[] ud
    uy:i32[16] <- ua[ux:ux+16]
    uz:i32[16] = add uw uy
    va:i32[] = add 1792:i32[] ud
    vb:i32[16] <- ua[va:va+16]
    vc:i32[16] = add uz vb
    vd:i32[] = add 2048:i32[] ud
    ve:i32[16] <- ua[vd:vd+16]
    vf:i32[16] = add vc ve
    vg:i32[] = add 2304:i32[] ud
    vh:i32[16] <- ua[vg:vg+16]
    vi:i32[16] = add vf vh
    vj:i32[] = add 2560:i32[] ud
    vk:i32[16] <- ua[vj:vj+16]
    vl:i32[16] = add vi vk
    vm:i32[] = add 2816:i32[] ud
    vn:i32[16] <- ua[vm:vm+16]
    vo:i32[16] = add vl vn
    vp:i32[] = add 3072:i32[] ud
    vq:i32[16] <- ua[vp:vp+16]
    vr:i32[16] = add vo vq
    vs:i32[] = add 3328:i32[] ud
    vt:i32[16] <- ua[vs:vs+16]
    vu:i32[16] = add vr vt
    vv:i32[] = add 3584:i32[] ud
    vw:i32[16] <- ua[vv:vv+16]
    vx:i32[16] = add vu vw
    vy:i32[] = add 3840:i32[] ud
    vz:i32[16] <- ua[vy:vy+16]
    wa:i32[16] = add vx vz
    wb:i32[16], ub[ud:ud+16] <- ub[ud:ud+16], wa
  in () } in
let jaxpr7 = { lambda wc:Ref<vmem>{f32[10240]} wd:i32[16] we:i32[] wf:i32[] wg:i32[]
    wh:i32[] wi:bool[] wj:Ref<vmem>{i32[2064]}; wk:i32[] wl:i32[] wm:i32[]. let
    wn:i32[] = mul wk 16:i32[]
    wo:f32[16] <- wc[wn:wn+16]
    wp:i32[16] = bitcast_convert_type[new_dtype=int32] wo
    wq:i32[] = mul wk 16:i32[]
    wr:i32[16] = add wq wd
    ws:bool[16] = lt wr we
    wt:bool[16] = gt wp -2139095041:i32[]
    wu:bool[16] = and ws wt
    wv:i32[16] = shift_right_arithmetic wp wf
    ww:bool[16] = gt wv wg
    wx:bool[16] = and wu ww
    wy:bool[16] = eq wv wg
    wz:bool[16] = and wu wy
    xa:i32[16] = convert_element_type[new_dtype=int32 weak_type=False] wz
    xb:bool[16] = broadcast_in_dim True:bool[]
    xc:i32[16] = masked_cumsum xa xb
    xd:i32[16] = add xc wm
    xe:bool[16] = le xd wh
    xf:bool[16] = and wz xe
    xg:bool[16] = or wx xf
    xh:bool[16] = jit[
      name=_where
      jaxpr={ lambda ; wi:bool[] wu:bool[16] xg:bool[16]. let
          xh:bool[16] = select_n wi xg wu
        in (xh,) }
    ] wi wu xg
    xi:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] wj wr wl xh
    xj:i32[16] = all_reduce_population_count[reduce=1] xh
    xk:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] xj
    xl:i32[] = squeeze[dimensions=(0,)] xk
    xm:i32[] = add wl xl
    xn:i32[1] = slice[limit_indices=(16,) start_indices=(15,) strides=None] xd
    xo:i32[] = squeeze[dimensions=(0,)] xn
  in (xm, xo) } in
let jaxpr8 = { lambda xp:Ref<vmem>{f32[10240]} xq:i32[16] xr:i32[] xs:Ref<vmem>{i32[4112]}; xt:i32[]. let
    xu:i32[] = mul xt 16:i32[]
    xv:f32[16] <- xp[xu:xu+16]
    xw:i32[16] = bitcast_convert_type[new_dtype=int32] xv
    xx:i32[] = mul xt 16:i32[]
    xy:i32[16] = add xx xq
    xz:bool[16] = lt xy xr
    ya:bool[16] = gt xw -2139095041:i32[]
    yb:bool[16] = and xz ya
    yc:i32[16] = shift_right_arithmetic xw 24:i32[]
    yd:i32[16] = add yc 128:i32[]
    ye:i32[16] = mul xq 256:i32[]
    yf:i32[16] = add ye yd
    yg:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] xs yf yg yb
  in () } in
let jaxpr9 = { lambda yh:Ref<vmem>{i32[4112]} yi:Ref<vmem>{i32[272]}; yj:i32[]. let
    yk:i32[] = mul yj 16:i32[]
    yl:i32[16] = broadcast_in_dim 0:i32[]
    ym:i32[] = add 0:i32[] yk
    yn:i32[16] <- yh[ym:ym+16]
    yo:i32[16] = add yl yn
    yp:i32[] = add 256:i32[] yk
    yq:i32[16] <- yh[yp:yp+16]
    yr:i32[16] = add yo yq
    ys:i32[] = add 512:i32[] yk
    yt:i32[16] <- yh[ys:ys+16]
    yu:i32[16] = add yr yt
    yv:i32[] = add 768:i32[] yk
    yw:i32[16] <- yh[yv:yv+16]
    yx:i32[16] = add yu yw
    yy:i32[] = add 1024:i32[] yk
    yz:i32[16] <- yh[yy:yy+16]
    za:i32[16] = add yx yz
    zb:i32[] = add 1280:i32[] yk
    zc:i32[16] <- yh[zb:zb+16]
    zd:i32[16] = add za zc
    ze:i32[] = add 1536:i32[] yk
    zf:i32[16] <- yh[ze:ze+16]
    zg:i32[16] = add zd zf
    zh:i32[] = add 1792:i32[] yk
    zi:i32[16] <- yh[zh:zh+16]
    zj:i32[16] = add zg zi
    zk:i32[] = add 2048:i32[] yk
    zl:i32[16] <- yh[zk:zk+16]
    zm:i32[16] = add zj zl
    zn:i32[] = add 2304:i32[] yk
    zo:i32[16] <- yh[zn:zn+16]
    zp:i32[16] = add zm zo
    zq:i32[] = add 2560:i32[] yk
    zr:i32[16] <- yh[zq:zq+16]
    zs:i32[16] = add zp zr
    zt:i32[] = add 2816:i32[] yk
    zu:i32[16] <- yh[zt:zt+16]
    zv:i32[16] = add zs zu
    zw:i32[] = add 3072:i32[] yk
    zx:i32[16] <- yh[zw:zw+16]
    zy:i32[16] = add zv zx
    zz:i32[] = add 3328:i32[] yk
    baa:i32[16] <- yh[zz:zz+16]
    bab:i32[16] = add zy baa
    bac:i32[] = add 3584:i32[] yk
    bad:i32[16] <- yh[bac:bac+16]
    bae:i32[16] = add bab bad
    baf:i32[] = add 3840:i32[] yk
    bag:i32[16] <- yh[baf:baf+16]
    bah:i32[16] = add bae bag
    bai:i32[16], yi[yk:yk+16] <- yi[yk:yk+16], bah
  in () } in
{ lambda ; baj:Ref<hbm>{f32[10485760]} bak:Ref<hbm>{i32[1024]} bal:Ref<hbm>{i32[2097152]}
    bam:Ref<vmem>{f32[10240]} ban:Ref<vmem>{i32[4112]} bao:Ref<vmem>{i32[272]} bap:Ref<vmem>{i32[2064]}
    baq:Ref<vmem>{i32[1024]} bar:Ref<semaphore_mem>{dma_sem[4]}. let
    bas:i32[] = axis_index[axis_name=core] 
    bat:i32[] = axis_index[axis_name=subcore] 
    bau:i32[] = mul bas 16:i32[]
    bav:i32[] = add bau bat
    baw:i32[16] = iota[dimension=0 dtype=int32 shape=(16,) sharding=None] 
    dma_start(p0) bak[:] -> baq[...] bar[0]
    dma_wait baq[...] bar[0]
    _:i32[] _:i32[] = scan[
      jaxpr=jaxpr
      length=32
      num_carry=2
      num_consts=10
      reverse=False
      unroll=1
    ] bav baq baj bam bar ban baw bao bap bal 0:i32[] 0:i32[]
  in () }

The kernel jaxpr idx=0 for mpmd_map sc_topk_direct_b1024_n10240_k2048_f32in at /mnt/pd/home-mp/vllm-torchtpu-test3/vllm-torchtpu/kernel_work/sparsecore-topk/variants/v12-f32-input/_dbg_variant.py:356:
let jaxpr = { lambda ; a:i32[] b:Ref<vmem>{i32[1024]} c:Ref<hbm>{f32[10485760]} d:Ref<vmem>{f32[10240]}
    e:Ref<semaphore_mem>{dma_sem[4]} f:Ref<vmem>{i32[4112]} g:i32[16] h:Ref<vmem>{i32[272]}
    i:Ref<vmem>{i32[2064]} j:Ref<hbm>{i32[2097152]} k:i32[] l:i32[]. let
    m:i32[] = add k 1:i32[]
    n:i32[] = mul k 32:i32[]
    o:i32[] = convert_element_type[new_dtype=int32 weak_type=False] n
    p:i32[] = add o a
    q:bool[] = lt p 1024:i32[]
    r:i32[] = min p 1023:i32[]
    s:i32[16] <- b[r:r+16]
    t:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] s
    u:i32[] = squeeze[dimensions=(0,)] t
    v:i32[] = min u 10240:i32[]
    w:i32[] = jit[name=_where jaxpr=_where] q v 0:i32[]
    x:i32[] = mul r 10240:i32[]
    y:bool[] = gt w 0:i32[]
    z:i32[] = min 10240:i32[] w
    ba:i32[] = jit[name=_where jaxpr=_where] y z 0:i32[]
    bb:i32[] = add ba 15:i32[]
    bc:i32[] = shift_right_arithmetic bb 4:i32[]
    bd:bool[] = gt ba 0:i32[]
    be:i32[] = convert_element_type[new_dtype=int32 weak_type=False] bd
    cond[
      branches=(
        { lambda ; bf:Ref<hbm>{f32[10485760]} bg:i32[] bh:Ref<vmem>{f32[10240]} bi:Ref<semaphore_mem>{dma_sem[4]}
            bj:i32[]. let
            
          in () }
        { lambda ; bk:Ref<hbm>{f32[10485760]} bl:i32[] bm:Ref<vmem>{f32[10240]} bn:Ref<semaphore_mem>{dma_sem[4]}
            bo:i32[]. let
            dma_start(p0) bk[bl:bl+10240] -> bm[...] bn[1]
            dma_wait bm[...] bn[1]
            parallel_loop[
              jaxpr={ lambda bp:Ref<vmem>{f32[10240]}; bq:i32[]. let
                  br:i32[] = mul bq 16:i32[]
                  bs:f32[16] <- bp[br:br+16]
                  bt:i32[16] = bitcast_convert_type[new_dtype=int32] bs
                  bu:i32[16] = shift_right_arithmetic bt 31:i32[]
                  bv:i32[16] = and bu 2147483647:i32[]
                  bw:i32[16] = xor bt bv
                  bx:f32[16] = bitcast_convert_type[new_dtype=float32] bw
                  by:i32[] = mul bq 16:i32[]
                  bz:f32[16], bp[by:by+16] <- bp[by:by+16], bx
                in () }
              tree=PyTreeDef((*, *, *, [*], []))
              unroll=4
            ] 0:i32[] bo 1:i32[] bm
          in () }
      )
    ] be c x d e bc
    ca:bool[] = gt ba 0:i32[]
    cb:i32[] = convert_element_type[new_dtype=int32 weak_type=False] ca
    cond[
      branches=(
        { lambda ; cc:Ref<vmem>{i32[4112]} cd:i32[] ce:Ref<vmem>{f32[10240]} cf:i32[16]
            cg:i32[] ch:Ref<vmem>{i32[272]}. let
            
          in () }
        { lambda ; ci:Ref<vmem>{i32[4112]} cj:i32[] ck:Ref<vmem>{f32[10240]} cl:i32[16]
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
              jaxpr=jaxpr8
              tree=PyTreeDef((*, *, *, [*, *, *, *], []))
              unroll=4
            ] 0:i32[] cj 1:i32[] ck cl cm ci
            parallel_loop[
              jaxpr=jaxpr9
              tree=PyTreeDef((*, *, *, [*, *], []))
              unroll=2
            ] 0:i32[] 16:i32[] 1:i32[] ci cn
          in () }
      )
    ] cb f bc d g ba h
    ct:bool[] = eq ba 0:i32[]
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
    et:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda eu:Ref<vmem>{i32[4112]}; ev:i32[]. let
          ew:i32[16] = broadcast_in_dim 0:i32[]
          ex:i32[] = mul ev 16:i32[]
          ey:i32[16], eu[ex:ex+16] <- eu[ex:ex+16], ew
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] et 1:i32[] f
    ez:i32[] = jit[name=_where jaxpr=_where2] es 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr1
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] ez 1:i32[] d g ba 24:i32[] eq f
    fa:i32[] = jit[name=_where jaxpr=_where1] es 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr2 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      fa 1:i32[] f h
    fb:i32[] = parallel_loop[
      jaxpr={ lambda fc:Ref<vmem>{i32[272]}; fd:i32[] fe:i32[]. let
          ff:i32[] = mul fd 16:i32[]
          fg:i32[16] <- fc[ff:ff+16]
          fh:bool[16] = broadcast_in_dim True:bool[]
          fi:i32[16] = masked_cumsum fg fh
          fj:i32[16] = add fi fe
          fk:i32[] = mul fd 16:i32[]
          fl:i32[16], fc[fk:fk+16] <- fc[fk:fk+16], fj
          fm:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] fj
          fn:i32[] = squeeze[dimensions=(0,)] fm
        in (fn,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] fa 1:i32[] h 0:i32[]
    fo:i32[] = sub fb ep
    fp:i32[] = parallel_loop[
      jaxpr={ lambda fq:Ref<vmem>{i32[272]} fr:i32[]; fs:i32[] ft:i32[]. let
          fu:i32[] = mul fs 16:i32[]
          fv:i32[16] <- fq[fu:fu+16]
          fw:bool[16] = le fv fr
          fx:i32[16] = all_reduce_population_count[reduce=1] fw
          fy:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] fx
          fz:i32[] = squeeze[dimensions=(0,)] fy
          ga:i32[] = add ft fz
        in (ga,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] fa 1:i32[] h fo 0:i32[]
    gb:i32[16] <- h[fp:fp+16]
    gc:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gb
    gd:i32[] = squeeze[dimensions=(0,)] gc
    ge:bool[] = gt fp 0:i32[]
    gf:i32[] = sub fp 1:i32[]
    gg:i32[] = max gf 0:i32[]
    gh:i32[16] <- h[gg:gg+16]
    gi:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] gh
    gj:i32[] = squeeze[dimensions=(0,)] gi
    gk:i32[] = jit[name=_where jaxpr=_where] ge gj 0:i32[]
    gl:i32[] = sub fb gd
    gm:i32[] = sub gd gk
    gn:i32[] = sub ep gl
    go:i32[] = sub 24:i32[] 16:i32[]
    gp:i32[] = shift_left eq go
    gq:i32[] = or gp fp
    gr:bool[] = eq gn gm
    gs:bool[] = or es gr
    gt:bool[] = eq gn 0:i32[]
    gu:bool[] = or gs gt
    gv:i32[] = jit[name=_where jaxpr=_where3] es eq gq
    gw:i32[] = jit[name=_where jaxpr=_where3] es 24:i32[] 16:i32[]
    gx:i32[] = add en gl
    gy:i32[] = jit[name=_where jaxpr=_where3] es en gx
    gz:i32[] = jit[name=_where jaxpr=_where3] es ep gn
    ha:i32[] = jit[name=_where jaxpr=_where3] es eo gm
    hb:i32[] = jit[name=_where jaxpr=_where1] gu 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda hc:Ref<vmem>{i32[4112]}; hd:i32[]. let
          he:i32[16] = broadcast_in_dim 0:i32[]
          hf:i32[] = mul hd 16:i32[]
          hg:i32[16], hc[hf:hf+16] <- hc[hf:hf+16], he
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] hb 1:i32[] f
    hh:i32[] = jit[name=_where jaxpr=_where2] gu 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr3
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] hh 1:i32[] d g ba gw gv f
    hi:i32[] = jit[name=_where jaxpr=_where1] gu 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr4 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      hi 1:i32[] f h
    hj:i32[] = parallel_loop[
      jaxpr={ lambda hk:Ref<vmem>{i32[272]}; hl:i32[] hm:i32[]. let
          hn:i32[] = mul hl 16:i32[]
          ho:i32[16] <- hk[hn:hn+16]
          hp:bool[16] = broadcast_in_dim True:bool[]
          hq:i32[16] = masked_cumsum ho hp
          hr:i32[16] = add hq hm
          hs:i32[] = mul hl 16:i32[]
          ht:i32[16], hk[hs:hs+16] <- hk[hs:hs+16], hr
          hu:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] hr
          hv:i32[] = squeeze[dimensions=(0,)] hu
        in (hv,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] hi 1:i32[] h 0:i32[]
    hw:i32[] = sub hj gz
    hx:i32[] = parallel_loop[
      jaxpr={ lambda hy:Ref<vmem>{i32[272]} hz:i32[]; ia:i32[] ib:i32[]. let
          ic:i32[] = mul ia 16:i32[]
          id:i32[16] <- hy[ic:ic+16]
          ie:bool[16] = le id hz
          if:i32[16] = all_reduce_population_count[reduce=1] ie
          ig:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] if
          ih:i32[] = squeeze[dimensions=(0,)] ig
          ii:i32[] = add ib ih
        in (ii,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] hi 1:i32[] h hw 0:i32[]
    ij:i32[16] <- h[hx:hx+16]
    ik:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ij
    il:i32[] = squeeze[dimensions=(0,)] ik
    im:bool[] = gt hx 0:i32[]
    in:i32[] = sub hx 1:i32[]
    io:i32[] = max in 0:i32[]
    ip:i32[16] <- h[io:io+16]
    iq:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] ip
    ir:i32[] = squeeze[dimensions=(0,)] iq
    is:i32[] = jit[name=_where jaxpr=_where] im ir 0:i32[]
    it:i32[] = sub hj il
    iu:i32[] = sub il is
    iv:i32[] = sub gz it
    iw:i32[] = sub gw 8:i32[]
    ix:i32[] = shift_left gv iw
    iy:i32[] = or ix hx
    iz:bool[] = eq iv iu
    ja:bool[] = or gu iz
    jb:bool[] = eq iv 0:i32[]
    jc:bool[] = or ja jb
    jd:i32[] = jit[name=_where jaxpr=_where3] gu gv iy
    je:i32[] = jit[name=_where jaxpr=_where3] gu gw 8:i32[]
    jf:i32[] = add gy it
    jg:i32[] = jit[name=_where jaxpr=_where3] gu gy jf
    jh:i32[] = jit[name=_where jaxpr=_where3] gu gz iv
    ji:i32[] = jit[name=_where jaxpr=_where3] gu ha iu
    jj:i32[] = jit[name=_where jaxpr=_where1] jc 0:i32[] 256:i32[]
    parallel_loop[
      jaxpr={ lambda jk:Ref<vmem>{i32[4112]}; jl:i32[]. let
          jm:i32[16] = broadcast_in_dim 0:i32[]
          jn:i32[] = mul jl 16:i32[]
          jo:i32[16], jk[jn:jn+16] <- jk[jn:jn+16], jm
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] jj 1:i32[] f
    jp:i32[] = jit[name=_where jaxpr=_where2] jc 0:i32[] bc
    parallel_loop[
      jaxpr=jaxpr5
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *], []))
      unroll=4
    ] 0:i32[] jp 1:i32[] d g ba je jd f
    jq:i32[] = jit[name=_where jaxpr=_where1] jc 0:i32[] 16:i32[]
    parallel_loop[jaxpr=jaxpr6 tree=PyTreeDef((*, *, *, [*, *], [])) unroll=2] 0:i32[]
      jq 1:i32[] f h
    jr:i32[] = parallel_loop[
      jaxpr={ lambda js:Ref<vmem>{i32[272]}; jt:i32[] ju:i32[]. let
          jv:i32[] = mul jt 16:i32[]
          jw:i32[16] <- js[jv:jv+16]
          jx:bool[16] = broadcast_in_dim True:bool[]
          jy:i32[16] = masked_cumsum jw jx
          jz:i32[16] = add jy ju
          ka:i32[] = mul jt 16:i32[]
          kb:i32[16], js[ka:ka+16] <- js[ka:ka+16], jz
          kc:i32[1] = slice[
            limit_indices=(16,)
            start_indices=(15,)
            strides=None
          ] jz
          kd:i32[] = squeeze[dimensions=(0,)] kc
        in (kd,) }
      tree=PyTreeDef((*, *, *, [*], [*]))
      unroll=2
    ] 0:i32[] jq 1:i32[] h 0:i32[]
    ke:i32[] = sub jr jh
    kf:i32[] = parallel_loop[
      jaxpr={ lambda kg:Ref<vmem>{i32[272]} kh:i32[]; ki:i32[] kj:i32[]. let
          kk:i32[] = mul ki 16:i32[]
          kl:i32[16] <- kg[kk:kk+16]
          km:bool[16] = le kl kh
          kn:i32[16] = all_reduce_population_count[reduce=1] km
          ko:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kn
          kp:i32[] = squeeze[dimensions=(0,)] ko
          kq:i32[] = add kj kp
        in (kq,) }
      tree=PyTreeDef((*, *, *, [*, *], [*]))
      unroll=2
    ] 0:i32[] jq 1:i32[] h ke 0:i32[]
    kr:i32[16] <- h[kf:kf+16]
    ks:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kr
    kt:i32[] = squeeze[dimensions=(0,)] ks
    ku:bool[] = gt kf 0:i32[]
    kv:i32[] = sub kf 1:i32[]
    kw:i32[] = max kv 0:i32[]
    kx:i32[16] <- h[kw:kw+16]
    ky:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] kx
    kz:i32[] = squeeze[dimensions=(0,)] ky
    la:i32[] = jit[name=_where jaxpr=_where] ku kz 0:i32[]
    lb:i32[] = sub jr kt
    lc:i32[] = sub kt la
    ld:i32[] = sub jh lb
    le:i32[] = sub je 0:i32[]
    lf:i32[] = shift_left jd le
    lg:i32[] = or lf kf
    lh:bool[] = eq ld lc
    li:bool[] = or jc lh
    lj:bool[] = eq ld 0:i32[]
    lk:bool[] = or li lj
    ll:i32[] = jit[name=_where jaxpr=_where3] jc jd lg
    lm:i32[] = jit[name=_where jaxpr=_where3] jc je 0:i32[]
    ln:i32[] = add jg lb
    lo:i32[] = jit[name=_where jaxpr=_where3] jc jg ln
    lp:i32[] = jit[name=_where jaxpr=_where3] jc jh ld
    lq:i32[] = jit[name=_where jaxpr=_where3] jc ji lc
    lr:i32[] = jit[name=_where jaxpr=_where2] dp 0:i32[] lp
    ls:bool[] = lt dc 2048:i32[]
    lt:bool[] = and q ls
    lu:i32[] = jit[name=_where jaxpr=_where1] lt 128:i32[] 0:i32[]
    parallel_loop[
      jaxpr={ lambda lv:Ref<vmem>{i32[2064]}; lw:i32[]. let
          lx:i32[16] = broadcast_in_dim -1:i32[]
          ly:i32[] = mul lw 16:i32[]
          lz:i32[16], lv[ly:ly+16] <- lv[ly:ly+16], lx
        in () }
      tree=PyTreeDef((*, *, *, [*], []))
      unroll=4
    ] 0:i32[] lu 1:i32[] i
    ma:i32[] = jit[name=_where jaxpr=_where] q bc 0:i32[]
    mb:i32[] mc:i32[] = parallel_loop[
      jaxpr=jaxpr7
      tree=PyTreeDef((*, *, *, [*, *, *, *, *, *, *, *], [*, *]))
      unroll=4
    ] 0:i32[] ma 1:i32[] d g ba lm ll lr dp i 0:i32[] 0:i32[]
    md:i32[] = convert_element_type[new_dtype=int32 weak_type=False] q
    cond[
      branches=(
        { lambda ; me:i32[] mf:Ref<vmem>{i32[2064]} mg:Ref<hbm>{i32[2097152]} mh:Ref<semaphore_mem>{dma_sem[4]}. let
            
          in () }
        { lambda ; mi:i32[] mj:Ref<vmem>{i32[2064]} mk:Ref<hbm>{i32[2097152]} ml:Ref<semaphore_mem>{dma_sem[4]}. let
            mm:i32[] = mul mi 2048:i32[]
            dma_start(p0) mj[:2048] -> mk[mm:mm+2048] ml[2]
            dma_wait mk[mm:mm+2048] ml[2]
          in () }
      )
    ] md r i j e
  in (m, 0:i32[]) } in
let _where = { lambda ; mn:bool[] mo:i32[] mp:i32[]. let
    mq:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mp
    mr:i32[] = select_n mn mq mo
  in (mr,) } in
let _where1 = { lambda ; ms:bool[] mt:i32[] mu:i32[]. let
    mv:i32[] = select_n ms mu mt
  in (mv,) } in
let _where2 = { lambda ; mw:bool[] mx:i32[] my:i32[]. let
    mz:i32[] = convert_element_type[new_dtype=int32 weak_type=False] mx
    na:i32[] = select_n mw my mz
  in (na,) } in
let jaxpr1 = { lambda nb:Ref<vmem>{f32[10240]} nc:i32[16] nd:i32[] ne:i32[] nf:i32[]
    ng:Ref<vmem>{i32[4112]}; nh:i32[]. let
    ni:i32[] = mul nh 16:i32[]
    nj:f32[16] <- nb[ni:ni+16]
    nk:i32[16] = bitcast_convert_type[new_dtype=int32] nj
    nl:i32[] = mul nh 16:i32[]
    nm:i32[16] = add nl nc
    nn:bool[16] = lt nm nd
    no:bool[16] = gt nk -2139095041:i32[]
    np:bool[16] = and nn no
    nq:i32[16] = shift_right_arithmetic nk ne
    nr:bool[16] = eq nq nf
    ns:bool[16] = and np nr
    nt:i32[16] = shift_right_arithmetic nk 16:i32[]
    nu:i32[16] = and nt 255:i32[]
    nv:i32[16] = mul nc 256:i32[]
    nw:i32[16] = add nv nu
    nx:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] ng nw nx ns
  in () } in
let jaxpr2 = { lambda ny:Ref<vmem>{i32[4112]} nz:Ref<vmem>{i32[272]}; oa:i32[]. let
    ob:i32[] = mul oa 16:i32[]
    oc:i32[16] = broadcast_in_dim 0:i32[]
    od:i32[] = add 0:i32[] ob
    oe:i32[16] <- ny[od:od+16]
    of:i32[16] = add oc oe
    og:i32[] = add 256:i32[] ob
    oh:i32[16] <- ny[og:og+16]
    oi:i32[16] = add of oh
    oj:i32[] = add 512:i32[] ob
    ok:i32[16] <- ny[oj:oj+16]
    ol:i32[16] = add oi ok
    om:i32[] = add 768:i32[] ob
    on:i32[16] <- ny[om:om+16]
    oo:i32[16] = add ol on
    op:i32[] = add 1024:i32[] ob
    oq:i32[16] <- ny[op:op+16]
    or:i32[16] = add oo oq
    os:i32[] = add 1280:i32[] ob
    ot:i32[16] <- ny[os:os+16]
    ou:i32[16] = add or ot
    ov:i32[] = add 1536:i32[] ob
    ow:i32[16] <- ny[ov:ov+16]
    ox:i32[16] = add ou ow
    oy:i32[] = add 1792:i32[] ob
    oz:i32[16] <- ny[oy:oy+16]
    pa:i32[16] = add ox oz
    pb:i32[] = add 2048:i32[] ob
    pc:i32[16] <- ny[pb:pb+16]
    pd:i32[16] = add pa pc
    pe:i32[] = add 2304:i32[] ob
    pf:i32[16] <- ny[pe:pe+16]
    pg:i32[16] = add pd pf
    ph:i32[] = add 2560:i32[] ob
    pi:i32[16] <- ny[ph:ph+16]
    pj:i32[16] = add pg pi
    pk:i32[] = add 2816:i32[] ob
    pl:i32[16] <- ny[pk:pk+16]
    pm:i32[16] = add pj pl
    pn:i32[] = add 3072:i32[] ob
    po:i32[16] <- ny[pn:pn+16]
    pp:i32[16] = add pm po
    pq:i32[] = add 3328:i32[] ob
    pr:i32[16] <- ny[pq:pq+16]
    ps:i32[16] = add pp pr
    pt:i32[] = add 3584:i32[] ob
    pu:i32[16] <- ny[pt:pt+16]
    pv:i32[16] = add ps pu
    pw:i32[] = add 3840:i32[] ob
    px:i32[16] <- ny[pw:pw+16]
    py:i32[16] = add pv px
    pz:i32[16], nz[ob:ob+16] <- nz[ob:ob+16], py
  in () } in
let _where3 = { lambda ; qa:bool[] qb:i32[] qc:i32[]. let
    qd:i32[] = select_n qa qc qb
  in (qd,) } in
let jaxpr3 = { lambda qe:Ref<vmem>{f32[10240]} qf:i32[16] qg:i32[] qh:i32[] qi:i32[]
    qj:Ref<vmem>{i32[4112]}; qk:i32[]. let
    ql:i32[] = mul qk 16:i32[]
    qm:f32[16] <- qe[ql:ql+16]
    qn:i32[16] = bitcast_convert_type[new_dtype=int32] qm
    qo:i32[] = mul qk 16:i32[]
    qp:i32[16] = add qo qf
    qq:bool[16] = lt qp qg
    qr:bool[16] = gt qn -2139095041:i32[]
    qs:bool[16] = and qq qr
    qt:i32[16] = shift_right_arithmetic qn qh
    qu:bool[16] = eq qt qi
    qv:bool[16] = and qs qu
    qw:i32[16] = shift_right_arithmetic qn 8:i32[]
    qx:i32[16] = and qw 255:i32[]
    qy:i32[16] = mul qf 256:i32[]
    qz:i32[16] = add qy qx
    ra:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] qj qz ra qv
  in () } in
let jaxpr4 = { lambda rb:Ref<vmem>{i32[4112]} rc:Ref<vmem>{i32[272]}; rd:i32[]. let
    re:i32[] = mul rd 16:i32[]
    rf:i32[16] = broadcast_in_dim 0:i32[]
    rg:i32[] = add 0:i32[] re
    rh:i32[16] <- rb[rg:rg+16]
    ri:i32[16] = add rf rh
    rj:i32[] = add 256:i32[] re
    rk:i32[16] <- rb[rj:rj+16]
    rl:i32[16] = add ri rk
    rm:i32[] = add 512:i32[] re
    rn:i32[16] <- rb[rm:rm+16]
    ro:i32[16] = add rl rn
    rp:i32[] = add 768:i32[] re
    rq:i32[16] <- rb[rp:rp+16]
    rr:i32[16] = add ro rq
    rs:i32[] = add 1024:i32[] re
    rt:i32[16] <- rb[rs:rs+16]
    ru:i32[16] = add rr rt
    rv:i32[] = add 1280:i32[] re
    rw:i32[16] <- rb[rv:rv+16]
    rx:i32[16] = add ru rw
    ry:i32[] = add 1536:i32[] re
    rz:i32[16] <- rb[ry:ry+16]
    sa:i32[16] = add rx rz
    sb:i32[] = add 1792:i32[] re
    sc:i32[16] <- rb[sb:sb+16]
    sd:i32[16] = add sa sc
    se:i32[] = add 2048:i32[] re
    sf:i32[16] <- rb[se:se+16]
    sg:i32[16] = add sd sf
    sh:i32[] = add 2304:i32[] re
    si:i32[16] <- rb[sh:sh+16]
    sj:i32[16] = add sg si
    sk:i32[] = add 2560:i32[] re
    sl:i32[16] <- rb[sk:sk+16]
    sm:i32[16] = add sj sl
    sn:i32[] = add 2816:i32[] re
    so:i32[16] <- rb[sn:sn+16]
    sp:i32[16] = add sm so
    sq:i32[] = add 3072:i32[] re
    sr:i32[16] <- rb[sq:sq+16]
    ss:i32[16] = add sp sr
    st:i32[] = add 3328:i32[] re
    su:i32[16] <- rb[st:st+16]
    sv:i32[16] = add ss su
    sw:i32[] = add 3584:i32[] re
    sx:i32[16] <- rb[sw:sw+16]
    sy:i32[16] = add sv sx
    sz:i32[] = add 3840:i32[] re
    ta:i32[16] <- rb[sz:sz+16]
    tb:i32[16] = add sy ta
    tc:i32[16], rc[re:re+16] <- rc[re:re+16], tb
  in () } in
let jaxpr5 = { lambda td:Ref<vmem>{f32[10240]} te:i32[16] tf:i32[] tg:i32[] th:i32[]
    ti:Ref<vmem>{i32[4112]}; tj:i32[]. let
    tk:i32[] = mul tj 16:i32[]
    tl:f32[16] <- td[tk:tk+16]
    tm:i32[16] = bitcast_convert_type[new_dtype=int32] tl
    tn:i32[] = mul tj 16:i32[]
    to:i32[16] = add tn te
    tp:bool[16] = lt to tf
    tq:bool[16] = gt tm -2139095041:i32[]
    tr:bool[16] = and tp tq
    ts:i32[16] = shift_right_arithmetic tm tg
    tt:bool[16] = eq ts th
    tu:bool[16] = and tr tt
    tv:i32[16] = shift_right_arithmetic tm 0:i32[]
    tw:i32[16] = and tv 255:i32[]
    tx:i32[16] = mul te 256:i32[]
    ty:i32[16] = add tx tw
    tz:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] ti ty tz tu
  in () } in
let jaxpr6 = { lambda ua:Ref<vmem>{i32[4112]} ub:Ref<vmem>{i32[272]}; uc:i32[]. let
    ud:i32[] = mul uc 16:i32[]
    ue:i32[16] = broadcast_in_dim 0:i32[]
    uf:i32[] = add 0:i32[] ud
    ug:i32[16] <- ua[uf:uf+16]
    uh:i32[16] = add ue ug
    ui:i32[] = add 256:i32[] ud
    uj:i32[16] <- ua[ui:ui+16]
    uk:i32[16] = add uh uj
    ul:i32[] = add 512:i32[] ud
    um:i32[16] <- ua[ul:ul+16]
    un:i32[16] = add uk um
    uo:i32[] = add 768:i32[] ud
    up:i32[16] <- ua[uo:uo+16]
    uq:i32[16] = add un up
    ur:i32[] = add 1024:i32[] ud
    us:i32[16] <- ua[ur:ur+16]
    ut:i32[16] = add uq us
    uu:i32[] = add 1280:i32[] ud
    uv:i32[16] <- ua[uu:uu+16]
    uw:i32[16] = add ut uv
    ux:i32[] = add 1536:i32[] ud
    uy:i32[16] <- ua[ux:ux+16]
    uz:i32[16] = add uw uy
    va:i32[] = add 1792:i32[] ud
    vb:i32[16] <- ua[va:va+16]
    vc:i32[16] = add uz vb
    vd:i32[] = add 2048:i32[] ud
    ve:i32[16] <- ua[vd:vd+16]
    vf:i32[16] = add vc ve
    vg:i32[] = add 2304:i32[] ud
    vh:i32[16] <- ua[vg:vg+16]
    vi:i32[16] = add vf vh
    vj:i32[] = add 2560:i32[] ud
    vk:i32[16] <- ua[vj:vj+16]
    vl:i32[16] = add vi vk
    vm:i32[] = add 2816:i32[] ud
    vn:i32[16] <- ua[vm:vm+16]
    vo:i32[16] = add vl vn
    vp:i32[] = add 3072:i32[] ud
    vq:i32[16] <- ua[vp:vp+16]
    vr:i32[16] = add vo vq
    vs:i32[] = add 3328:i32[] ud
    vt:i32[16] <- ua[vs:vs+16]
    vu:i32[16] = add vr vt
    vv:i32[] = add 3584:i32[] ud
    vw:i32[16] <- ua[vv:vv+16]
    vx:i32[16] = add vu vw
    vy:i32[] = add 3840:i32[] ud
    vz:i32[16] <- ua[vy:vy+16]
    wa:i32[16] = add vx vz
    wb:i32[16], ub[ud:ud+16] <- ub[ud:ud+16], wa
  in () } in
let jaxpr7 = { lambda wc:Ref<vmem>{f32[10240]} wd:i32[16] we:i32[] wf:i32[] wg:i32[]
    wh:i32[] wi:bool[] wj:Ref<vmem>{i32[2064]}; wk:i32[] wl:i32[] wm:i32[]. let
    wn:i32[] = mul wk 16:i32[]
    wo:f32[16] <- wc[wn:wn+16]
    wp:i32[16] = bitcast_convert_type[new_dtype=int32] wo
    wq:i32[] = mul wk 16:i32[]
    wr:i32[16] = add wq wd
    ws:bool[16] = lt wr we
    wt:bool[16] = gt wp -2139095041:i32[]
    wu:bool[16] = and ws wt
    wv:i32[16] = shift_right_arithmetic wp wf
    ww:bool[16] = gt wv wg
    wx:bool[16] = and wu ww
    wy:bool[16] = eq wv wg
    wz:bool[16] = and wu wy
    xa:i32[16] = convert_element_type[new_dtype=int32 weak_type=False] wz
    xb:bool[16] = broadcast_in_dim True:bool[]
    xc:i32[16] = masked_cumsum xa xb
    xd:i32[16] = add xc wm
    xe:bool[16] = le xd wh
    xf:bool[16] = and wz xe
    xg:bool[16] = or wx xf
    xh:bool[16] = jit[
      name=_where
      jaxpr={ lambda ; wi:bool[] wu:bool[16] xg:bool[16]. let
          xh:bool[16] = select_n wi xg wu
        in (xh,) }
    ] wi wu xg
    xi:i32[16] = swap[
      add=False
      has_mask=True
      tree=PyTreeDef((CustomNode(NDIndexer[(PyTreeDef((CustomNode(Slice[(None, 16, 1)], [*, None]),)), (2064,), ())], [*]),))
    ] wj wr wl xh
    xj:i32[16] = all_reduce_population_count[reduce=1] xh
    xk:i32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] xj
    xl:i32[] = squeeze[dimensions=(0,)] xk
    xm:i32[] = add wl xl
    xn:i32[1] = slice[limit_indices=(16,) start_indices=(15,) strides=None] xd
    xo:i32[] = squeeze[dimensions=(0,)] xn
  in (xm, xo) } in
let jaxpr8 = { lambda xp:Ref<vmem>{f32[10240]} xq:i32[16] xr:i32[] xs:Ref<vmem>{i32[4112]}; xt:i32[]. let
    xu:i32[] = mul xt 16:i32[]
    xv:f32[16] <- xp[xu:xu+16]
    xw:i32[16] = bitcast_convert_type[new_dtype=int32] xv
    xx:i32[] = mul xt 16:i32[]
    xy:i32[16] = add xx xq
    xz:bool[16] = lt xy xr
    ya:bool[16] = gt xw -2139095041:i32[]
    yb:bool[16] = and xz ya
    yc:i32[16] = shift_right_arithmetic xw 24:i32[]
    yd:i32[16] = add yc 128:i32[]
    ye:i32[16] = mul xq 256:i32[]
    yf:i32[16] = add ye yd
    yg:i32[16] = broadcast_in_dim 1:i32[]
    scatter[add=True tree=PyTreeDef((*, (), (*,), *, *))] xs yf yg yb
  in () } in
let jaxpr9 = { lambda yh:Ref<vmem>{i32[4112]} yi:Ref<vmem>{i32[272]}; yj:i32[]. let
    yk:i32[] = mul yj 16:i32[]
    yl:i32[16] = broadcast_in_dim 0:i32[]
    ym:i32[] = add 0:i32[] yk
    yn:i32[16] <- yh[ym:ym+16]
    yo:i32[16] = add yl yn
    yp:i32[] = add 256:i32[] yk
    yq:i32[16] <- yh[yp:yp+16]
    yr:i32[16] = add yo yq
    ys:i32[] = add 512:i32[] yk
    yt:i32[16] <- yh[ys:ys+16]
    yu:i32[16] = add yr yt
    yv:i32[] = add 768:i32[] yk
    yw:i32[16] <- yh[yv:yv+16]
    yx:i32[16] = add yu yw
    yy:i32[] = add 1024:i32[] yk
    yz:i32[16] <- yh[yy:yy+16]
    za:i32[16] = add yx yz
    zb:i32[] = add 1280:i32[] yk
    zc:i32[16] <- yh[zb:zb+16]
    zd:i32[16] = add za zc
    ze:i32[] = add 1536:i32[] yk
    zf:i32[16] <- yh[ze:ze+16]
    zg:i32[16] = add zd zf
    zh:i32[] = add 1792:i32[] yk
    zi:i32[16] <- yh[zh:zh+16]
    zj:i32[16] = add zg zi
    zk:i32[] = add 2048:i32[] yk
    zl:i32[16] <- yh[zk:zk+16]
    zm:i32[16] = add zj zl
    zn:i32[] = add 2304:i32[] yk
    zo:i32[16] <- yh[zn:zn+16]
    zp:i32[16] = add zm zo
    zq:i32[] = add 2560:i32[] yk
    zr:i32[16] <- yh[zq:zq+16]
    zs:i32[16] = add zp zr
    zt:i32[] = add 2816:i32[] yk
    zu:i32[16] <- yh[zt:zt+16]
    zv:i32[16] = add zs zu
    zw:i32[] = add 3072:i32[] yk
    zx:i32[16] <- yh[zw:zw+16]
    zy:i32[16] = add zv zx
    zz:i32[] = add 3328:i32[] yk
    baa:i32[16] <- yh[zz:zz+16]
    bab:i32[16] = add zy baa
    bac:i32[] = add 3584:i32[] yk
    bad:i32[16] <- yh[bac:bac+16]
    bae:i32[16] = add bab bad
    baf:i32[] = add 3840:i32[] yk
    bag:i32[16] <- yh[baf:baf+16]
    bah:i32[16] = add bae bag
    bai:i32[16], yi[yk:yk+16] <- yi[yk:yk+16], bah
  in () } in
{ lambda ; baj:Ref<hbm>{f32[10485760]} bak:Ref<hbm>{i32[1024]} bal:Ref<hbm>{i32[2097152]}
    bam:Ref<vmem>{f32[10240]} ban:Ref<vmem>{i32[4112]} bao:Ref<vmem>{i32[272]} bap:Ref<vmem>{i32[2064]}
    baq:Ref<vmem>{i32[1024]} bar:Ref<semaphore_mem>{dma_sem[4]}. let
    bas:i32[] = axis_index[axis_name=core] 
    bat:i32[] = axis_index[axis_name=subcore] 
    bau:i32[] = mul bas 16:i32[]
    bav:i32[] = add bau bat
    baw:i32[16] = iota[dimension=0 dtype=int32 shape=(16,) sharding=None] 
    dma_start(p0) bak[:] -> baq[...] bar[0]
    dma_wait baq[...] bar[0]
    _:i32[] _:i32[] = scan[
      jaxpr=jaxpr
      length=32
      num_carry=2
      num_consts=10
      reverse=False
      unroll=1
    ] bav baq baj bam bar ban baw bao bap bal 0:i32[] 0:i32[]
  in () }

The Mosaic module for mpmd_map:
#map = affine_map<(d0, d1) -> (0)>
module {
  func.func @sc_topk_direct_b1024_n10240_k2048_f32in(%arg0: i32, %arg1: i32, %arg2: memref<10485760xf32, #tpu.memory_space<hbm>>, %arg3: memref<1024xi32, #tpu.memory_space<hbm>>, %arg4: memref<2097152xi32, #tpu.memory_space<hbm>>, %arg5: memref<10240xf32, #tpu.memory_space<vmem>>, %arg6: memref<4112xi32, #tpu.memory_space<vmem>>, %arg7: memref<272xi32, #tpu.memory_space<vmem>>, %arg8: memref<2064xi32, #tpu.memory_space<vmem>>, %arg9: memref<1024xi32, #tpu.memory_space<vmem>>, %arg10: memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<core_parallel>, #tpu.dimension_semantics<subcore_parallel>], iteration_bounds = array<i64: 2, 16>, scalar_prefetch = 0 : i64, scratch_operands = 6 : i64, tpu.core_type = #tpu.core_type<sc_vector_subcore>, window_params = [{transform_indices = #map}, {transform_indices = #map}, {transform_indices = #map}]} {
    %cst = arith.constant dense<-1> : vector<16xi32>
    %cst_0 = arith.constant dense<8> : vector<16xi32>
    %cst_1 = arith.constant dense<255> : vector<16xi32>
    %cst_2 = arith.constant dense<16> : vector<16xi32>
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
    %3 = tpu.memref_slice %arg10[%c0_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %4 = tpu.memref_squeeze %3 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    tpu.enqueue_dma source(%arg3 : memref<1024xi32, #tpu.memory_space<hbm>>) target(%arg9 : memref<1024xi32, #tpu.memory_space<vmem>>) target_semaphore(%4 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    %5 = tpu.memref_slice %arg10[%c0_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    %6 = tpu.memref_squeeze %5 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
    tpu.wait_dma2 semaphore(%6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%arg3 : memref<1024xi32, #tpu.memory_space<hbm>>) dst(%arg9 : memref<1024xi32, #tpu.memory_space<vmem>>)
    scf.for %arg11 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %7 = arith.muli %arg11, %c32_i32 : i32
      %8 = arith.addi %7, %1 : i32
      %9 = arith.cmpi slt, %8, %c1024_i32 : i32
      %10 = arith.minsi %8, %c1023_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = tpu.vector_load %arg9[%11] {strides = array<i32>} : memref<1024xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
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
        %124 = tpu.memref_slice %arg10[%c1_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %125 = tpu.memref_squeeze %124 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %126 = tpu.memref_slice %arg2[%16] : memref<10485760xf32, #tpu.memory_space<hbm>> -> memref<10240xf32, #tpu.memory_space<hbm>>
        tpu.enqueue_dma source(%126 : memref<10240xf32, #tpu.memory_space<hbm>>) target(%arg5 : memref<10240xf32, #tpu.memory_space<vmem>>) target_semaphore(%125 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
        %127 = tpu.memref_slice %arg10[%c1_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %128 = tpu.memref_squeeze %127 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %129 = tpu.memref_slice %arg2[%16] : memref<10485760xf32, #tpu.memory_space<hbm>> -> memref<10240xf32, #tpu.memory_space<hbm>>
        tpu.wait_dma2 semaphore(%128 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%129 : memref<10240xf32, #tpu.memory_space<hbm>>) dst(%arg5 : memref<10240xf32, #tpu.memory_space<vmem>>)
        scf.for %arg12 = %c0_i32 to %21 step %c1_i32  : i32 {
          %130 = arith.muli %arg12, %c16_i32 : i32
          %131 = arith.index_cast %130 : i32 to index
          %132 = tpu.vector_load %arg5[%131] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
          %133 = tpu.bitcast %132 : vector<16xf32> -> vector<16xi32>
          %134 = arith.shrsi %133, %cst_11 : vector<16xi32>
          %135 = arith.andi %134, %cst_10 : vector<16xi32>
          %136 = arith.xori %133, %135 : vector<16xi32>
          %137 = tpu.bitcast %136 : vector<16xi32> -> vector<16xf32>
          %138 = arith.muli %arg12, %c16_i32 : i32
          %139 = arith.index_cast %138 : i32 to index
          tpu.vector_store %arg5[%139], %137 {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      }
      %23 = arith.cmpi sgt, %19, %c0_i32 : i32
      scf.if %23 {
        scf.for %arg12 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
          %124 = arith.muli %arg12, %c16_i32 : i32
          %125 = arith.index_cast %124 : i32 to index
          tpu.vector_store %arg6[%125], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
        scf.for %arg12 = %c0_i32 to %21 step %c1_i32  : i32 {
          %124 = arith.muli %arg12, %c16_i32 : i32
          %125 = arith.index_cast %124 : i32 to index
          %126 = tpu.vector_load %arg5[%125] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
          %127 = tpu.bitcast %126 : vector<16xf32> -> vector<16xi32>
          %128 = arith.muli %arg12, %c16_i32 : i32
          %129 = vector.broadcast %128 : i32 to vector<16xi32>
          %130 = arith.addi %129, %2 : vector<16xi32>
          %131 = vector.broadcast %19 : i32 to vector<16xi32>
          %132 = arith.cmpi slt, %130, %131 : vector<16xi32>
          %133 = arith.cmpi sgt, %127, %cst_8 : vector<16xi32>
          %134 = arith.andi %132, %133 : vector<16xi1>
          %135 = arith.shrsi %127, %cst_7 : vector<16xi32>
          %136 = arith.addi %135, %cst_6 : vector<16xi32>
          %137 = arith.muli %2, %cst_5 : vector<16xi32>
          %138 = arith.addi %137, %136 : vector<16xi32>
          tpu.vector_store_idx %arg6[%138], %cst_4 masked %134 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
        scf.for %arg12 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %124 = arith.muli %arg12, %c16_i32 : i32
          %125 = arith.index_cast %124 : i32 to index
          %126 = tpu.vector_load %arg6[%125] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %127 = arith.addi %124, %c256_i32 : i32
          %128 = arith.index_cast %127 : i32 to index
          %129 = tpu.vector_load %arg6[%128] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %130 = arith.addi %126, %129 : vector<16xi32>
          %131 = arith.addi %124, %c512_i32 : i32
          %132 = arith.index_cast %131 : i32 to index
          %133 = tpu.vector_load %arg6[%132] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %134 = arith.addi %130, %133 : vector<16xi32>
          %135 = arith.addi %124, %c768_i32 : i32
          %136 = arith.index_cast %135 : i32 to index
          %137 = tpu.vector_load %arg6[%136] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %138 = arith.addi %134, %137 : vector<16xi32>
          %139 = arith.addi %124, %c1024_i32 : i32
          %140 = arith.index_cast %139 : i32 to index
          %141 = tpu.vector_load %arg6[%140] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %142 = arith.addi %138, %141 : vector<16xi32>
          %143 = arith.addi %124, %c1280_i32 : i32
          %144 = arith.index_cast %143 : i32 to index
          %145 = tpu.vector_load %arg6[%144] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %146 = arith.addi %142, %145 : vector<16xi32>
          %147 = arith.addi %124, %c1536_i32 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = tpu.vector_load %arg6[%148] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %150 = arith.addi %146, %149 : vector<16xi32>
          %151 = arith.addi %124, %c1792_i32 : i32
          %152 = arith.index_cast %151 : i32 to index
          %153 = tpu.vector_load %arg6[%152] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %154 = arith.addi %150, %153 : vector<16xi32>
          %155 = arith.addi %124, %c2048_i32 : i32
          %156 = arith.index_cast %155 : i32 to index
          %157 = tpu.vector_load %arg6[%156] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %158 = arith.addi %154, %157 : vector<16xi32>
          %159 = arith.addi %124, %c2304_i32 : i32
          %160 = arith.index_cast %159 : i32 to index
          %161 = tpu.vector_load %arg6[%160] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %162 = arith.addi %158, %161 : vector<16xi32>
          %163 = arith.addi %124, %c2560_i32 : i32
          %164 = arith.index_cast %163 : i32 to index
          %165 = tpu.vector_load %arg6[%164] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %166 = arith.addi %162, %165 : vector<16xi32>
          %167 = arith.addi %124, %c2816_i32 : i32
          %168 = arith.index_cast %167 : i32 to index
          %169 = tpu.vector_load %arg6[%168] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %170 = arith.addi %166, %169 : vector<16xi32>
          %171 = arith.addi %124, %c3072_i32 : i32
          %172 = arith.index_cast %171 : i32 to index
          %173 = tpu.vector_load %arg6[%172] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %174 = arith.addi %170, %173 : vector<16xi32>
          %175 = arith.addi %124, %c3328_i32 : i32
          %176 = arith.index_cast %175 : i32 to index
          %177 = tpu.vector_load %arg6[%176] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %178 = arith.addi %174, %177 : vector<16xi32>
          %179 = arith.addi %124, %c3584_i32 : i32
          %180 = arith.index_cast %179 : i32 to index
          %181 = tpu.vector_load %arg6[%180] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %182 = arith.addi %178, %181 : vector<16xi32>
          %183 = arith.addi %124, %c3840_i32 : i32
          %184 = arith.index_cast %183 : i32 to index
          %185 = tpu.vector_load %arg6[%184] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
          %186 = arith.addi %182, %185 : vector<16xi32>
          %187 = arith.index_cast %124 : i32 to index
          tpu.vector_store %arg7[%187], %186 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      }
      %24 = arith.cmpi eq, %19, %c0_i32 : i32
      scf.if %24 {
        scf.for %arg12 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %124 = arith.muli %arg12, %c16_i32 : i32
          %125 = arith.index_cast %124 : i32 to index
          tpu.vector_store %arg7[%125], %cst_9 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      }
      %25 = scf.for %arg12 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = tpu.scan <sum>, %126 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %128 = vector.broadcast %arg13 : i32 to vector<16xi32>
        %129 = arith.addi %127, %128 : vector<16xi32>
        %130 = arith.muli %arg12, %c16_i32 : i32
        %131 = arith.index_cast %130 : i32 to index
        tpu.vector_store %arg7[%131], %129 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %132 = vector.extract %129[15] : i32 from vector<16xi32>
        scf.yield %132 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %26 = arith.cmpi sle, %25, %c2048_i32 : i32
      %27 = arith.subi %25, %c2048_i32 : i32
      %28 = scf.for %arg12 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = vector.broadcast %27 : i32 to vector<16xi32>
        %128 = arith.cmpi sle, %126, %127 : vector<16xi32>
        %129 = tpu.all_reduce %128 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %130 = vector.extract %129[0] : i32 from vector<16xi32>
        %131 = arith.addi %arg13, %130 : i32
        scf.yield %131 : i32
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
      %45 = arith.select %44, %c0_i32, %c256_i32 : i32
      scf.for %arg12 = %c0_i32 to %45 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg6[%125], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %46 = arith.select %44, %c0_i32, %21 : i32
      scf.for %arg12 = %c0_i32 to %46 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg5[%125] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
        %127 = tpu.bitcast %126 : vector<16xf32> -> vector<16xi32>
        %128 = arith.muli %arg12, %c16_i32 : i32
        %129 = vector.broadcast %128 : i32 to vector<16xi32>
        %130 = arith.addi %129, %2 : vector<16xi32>
        %131 = vector.broadcast %19 : i32 to vector<16xi32>
        %132 = arith.cmpi slt, %130, %131 : vector<16xi32>
        %133 = arith.cmpi sgt, %127, %cst_8 : vector<16xi32>
        %134 = arith.andi %132, %133 : vector<16xi1>
        %135 = arith.shrsi %127, %cst_7 : vector<16xi32>
        %136 = vector.broadcast %42 : i32 to vector<16xi32>
        %137 = arith.cmpi eq, %135, %136 : vector<16xi32>
        %138 = arith.andi %134, %137 : vector<16xi1>
        %139 = arith.shrsi %127, %cst_2 : vector<16xi32>
        %140 = arith.andi %139, %cst_1 : vector<16xi32>
        %141 = arith.muli %2, %cst_5 : vector<16xi32>
        %142 = arith.addi %141, %140 : vector<16xi32>
        tpu.vector_store_idx %arg6[%142], %cst_4 masked %138 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %47 = arith.select %44, %c0_i32, %c16_i32 : i32
      scf.for %arg12 = %c0_i32 to %47 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg6[%125] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = arith.addi %124, %c256_i32 : i32
        %128 = arith.index_cast %127 : i32 to index
        %129 = tpu.vector_load %arg6[%128] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %130 = arith.addi %126, %129 : vector<16xi32>
        %131 = arith.addi %124, %c512_i32 : i32
        %132 = arith.index_cast %131 : i32 to index
        %133 = tpu.vector_load %arg6[%132] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %134 = arith.addi %130, %133 : vector<16xi32>
        %135 = arith.addi %124, %c768_i32 : i32
        %136 = arith.index_cast %135 : i32 to index
        %137 = tpu.vector_load %arg6[%136] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %138 = arith.addi %134, %137 : vector<16xi32>
        %139 = arith.addi %124, %c1024_i32 : i32
        %140 = arith.index_cast %139 : i32 to index
        %141 = tpu.vector_load %arg6[%140] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %142 = arith.addi %138, %141 : vector<16xi32>
        %143 = arith.addi %124, %c1280_i32 : i32
        %144 = arith.index_cast %143 : i32 to index
        %145 = tpu.vector_load %arg6[%144] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %146 = arith.addi %142, %145 : vector<16xi32>
        %147 = arith.addi %124, %c1536_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        %149 = tpu.vector_load %arg6[%148] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %150 = arith.addi %146, %149 : vector<16xi32>
        %151 = arith.addi %124, %c1792_i32 : i32
        %152 = arith.index_cast %151 : i32 to index
        %153 = tpu.vector_load %arg6[%152] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %154 = arith.addi %150, %153 : vector<16xi32>
        %155 = arith.addi %124, %c2048_i32 : i32
        %156 = arith.index_cast %155 : i32 to index
        %157 = tpu.vector_load %arg6[%156] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %158 = arith.addi %154, %157 : vector<16xi32>
        %159 = arith.addi %124, %c2304_i32 : i32
        %160 = arith.index_cast %159 : i32 to index
        %161 = tpu.vector_load %arg6[%160] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %162 = arith.addi %158, %161 : vector<16xi32>
        %163 = arith.addi %124, %c2560_i32 : i32
        %164 = arith.index_cast %163 : i32 to index
        %165 = tpu.vector_load %arg6[%164] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %166 = arith.addi %162, %165 : vector<16xi32>
        %167 = arith.addi %124, %c2816_i32 : i32
        %168 = arith.index_cast %167 : i32 to index
        %169 = tpu.vector_load %arg6[%168] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %170 = arith.addi %166, %169 : vector<16xi32>
        %171 = arith.addi %124, %c3072_i32 : i32
        %172 = arith.index_cast %171 : i32 to index
        %173 = tpu.vector_load %arg6[%172] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %174 = arith.addi %170, %173 : vector<16xi32>
        %175 = arith.addi %124, %c3328_i32 : i32
        %176 = arith.index_cast %175 : i32 to index
        %177 = tpu.vector_load %arg6[%176] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %178 = arith.addi %174, %177 : vector<16xi32>
        %179 = arith.addi %124, %c3584_i32 : i32
        %180 = arith.index_cast %179 : i32 to index
        %181 = tpu.vector_load %arg6[%180] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %182 = arith.addi %178, %181 : vector<16xi32>
        %183 = arith.addi %124, %c3840_i32 : i32
        %184 = arith.index_cast %183 : i32 to index
        %185 = tpu.vector_load %arg6[%184] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %186 = arith.addi %182, %185 : vector<16xi32>
        %187 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg7[%187], %186 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %48 = scf.for %arg12 = %c0_i32 to %47 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = tpu.scan <sum>, %126 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %128 = vector.broadcast %arg13 : i32 to vector<16xi32>
        %129 = arith.addi %127, %128 : vector<16xi32>
        %130 = arith.muli %arg12, %c16_i32 : i32
        %131 = arith.index_cast %130 : i32 to index
        tpu.vector_store %arg7[%131], %129 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %132 = vector.extract %129[15] : i32 from vector<16xi32>
        scf.yield %132 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %49 = arith.subi %48, %41 : i32
      %50 = scf.for %arg12 = %c0_i32 to %47 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = vector.broadcast %49 : i32 to vector<16xi32>
        %128 = arith.cmpi sle, %126, %127 : vector<16xi32>
        %129 = tpu.all_reduce %128 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %130 = vector.extract %129[0] : i32 from vector<16xi32>
        %131 = arith.addi %arg13, %130 : i32
        scf.yield %131 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %51 = arith.index_cast %50 : i32 to index
      %52 = tpu.vector_load %arg7[%51] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %53 = vector.extract %52[0] : i32 from vector<16xi32>
      %54 = arith.cmpi sgt, %50, %c0_i32 : i32
      %55 = arith.subi %50, %c1_i32 : i32
      %56 = arith.maxsi %55, %c0_i32 : i32
      %57 = arith.index_cast %56 : i32 to index
      %58 = tpu.vector_load %arg7[%57] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %59 = vector.extract %58[0] : i32 from vector<16xi32>
      %60 = arith.select %54, %59, %c0_i32 : i32
      %61 = arith.subi %48, %53 : i32
      %62 = arith.subi %53, %60 : i32
      %63 = arith.subi %41, %61 : i32
      %64 = arith.shli %42, %c8_i32 : i32
      %65 = arith.ori %64, %50 : i32
      %66 = arith.cmpi eq, %63, %62 : i32
      %67 = arith.ori %44, %66 : i1
      %68 = arith.cmpi eq, %63, %c0_i32 : i32
      %69 = arith.ori %67, %68 : i1
      %70 = arith.select %44, %42, %65 : i32
      %71 = arith.select %44, %c24_i32, %c16_i32 : i32
      %72 = arith.select %44, %41, %63 : i32
      %73 = arith.select %69, %c0_i32, %c256_i32 : i32
      scf.for %arg12 = %c0_i32 to %73 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg6[%125], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %74 = arith.select %69, %c0_i32, %21 : i32
      scf.for %arg12 = %c0_i32 to %74 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg5[%125] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
        %127 = tpu.bitcast %126 : vector<16xf32> -> vector<16xi32>
        %128 = arith.muli %arg12, %c16_i32 : i32
        %129 = vector.broadcast %128 : i32 to vector<16xi32>
        %130 = arith.addi %129, %2 : vector<16xi32>
        %131 = vector.broadcast %19 : i32 to vector<16xi32>
        %132 = arith.cmpi slt, %130, %131 : vector<16xi32>
        %133 = arith.cmpi sgt, %127, %cst_8 : vector<16xi32>
        %134 = arith.andi %132, %133 : vector<16xi1>
        %135 = vector.broadcast %71 : i32 to vector<16xi32>
        %136 = arith.shrsi %127, %135 : vector<16xi32>
        %137 = vector.broadcast %70 : i32 to vector<16xi32>
        %138 = arith.cmpi eq, %136, %137 : vector<16xi32>
        %139 = arith.andi %134, %138 : vector<16xi1>
        %140 = arith.shrsi %127, %cst_0 : vector<16xi32>
        %141 = arith.andi %140, %cst_1 : vector<16xi32>
        %142 = arith.muli %2, %cst_5 : vector<16xi32>
        %143 = arith.addi %142, %141 : vector<16xi32>
        tpu.vector_store_idx %arg6[%143], %cst_4 masked %139 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %75 = arith.select %69, %c0_i32, %c16_i32 : i32
      scf.for %arg12 = %c0_i32 to %75 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg6[%125] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = arith.addi %124, %c256_i32 : i32
        %128 = arith.index_cast %127 : i32 to index
        %129 = tpu.vector_load %arg6[%128] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %130 = arith.addi %126, %129 : vector<16xi32>
        %131 = arith.addi %124, %c512_i32 : i32
        %132 = arith.index_cast %131 : i32 to index
        %133 = tpu.vector_load %arg6[%132] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %134 = arith.addi %130, %133 : vector<16xi32>
        %135 = arith.addi %124, %c768_i32 : i32
        %136 = arith.index_cast %135 : i32 to index
        %137 = tpu.vector_load %arg6[%136] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %138 = arith.addi %134, %137 : vector<16xi32>
        %139 = arith.addi %124, %c1024_i32 : i32
        %140 = arith.index_cast %139 : i32 to index
        %141 = tpu.vector_load %arg6[%140] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %142 = arith.addi %138, %141 : vector<16xi32>
        %143 = arith.addi %124, %c1280_i32 : i32
        %144 = arith.index_cast %143 : i32 to index
        %145 = tpu.vector_load %arg6[%144] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %146 = arith.addi %142, %145 : vector<16xi32>
        %147 = arith.addi %124, %c1536_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        %149 = tpu.vector_load %arg6[%148] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %150 = arith.addi %146, %149 : vector<16xi32>
        %151 = arith.addi %124, %c1792_i32 : i32
        %152 = arith.index_cast %151 : i32 to index
        %153 = tpu.vector_load %arg6[%152] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %154 = arith.addi %150, %153 : vector<16xi32>
        %155 = arith.addi %124, %c2048_i32 : i32
        %156 = arith.index_cast %155 : i32 to index
        %157 = tpu.vector_load %arg6[%156] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %158 = arith.addi %154, %157 : vector<16xi32>
        %159 = arith.addi %124, %c2304_i32 : i32
        %160 = arith.index_cast %159 : i32 to index
        %161 = tpu.vector_load %arg6[%160] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %162 = arith.addi %158, %161 : vector<16xi32>
        %163 = arith.addi %124, %c2560_i32 : i32
        %164 = arith.index_cast %163 : i32 to index
        %165 = tpu.vector_load %arg6[%164] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %166 = arith.addi %162, %165 : vector<16xi32>
        %167 = arith.addi %124, %c2816_i32 : i32
        %168 = arith.index_cast %167 : i32 to index
        %169 = tpu.vector_load %arg6[%168] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %170 = arith.addi %166, %169 : vector<16xi32>
        %171 = arith.addi %124, %c3072_i32 : i32
        %172 = arith.index_cast %171 : i32 to index
        %173 = tpu.vector_load %arg6[%172] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %174 = arith.addi %170, %173 : vector<16xi32>
        %175 = arith.addi %124, %c3328_i32 : i32
        %176 = arith.index_cast %175 : i32 to index
        %177 = tpu.vector_load %arg6[%176] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %178 = arith.addi %174, %177 : vector<16xi32>
        %179 = arith.addi %124, %c3584_i32 : i32
        %180 = arith.index_cast %179 : i32 to index
        %181 = tpu.vector_load %arg6[%180] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %182 = arith.addi %178, %181 : vector<16xi32>
        %183 = arith.addi %124, %c3840_i32 : i32
        %184 = arith.index_cast %183 : i32 to index
        %185 = tpu.vector_load %arg6[%184] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %186 = arith.addi %182, %185 : vector<16xi32>
        %187 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg7[%187], %186 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %76 = scf.for %arg12 = %c0_i32 to %75 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = tpu.scan <sum>, %126 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %128 = vector.broadcast %arg13 : i32 to vector<16xi32>
        %129 = arith.addi %127, %128 : vector<16xi32>
        %130 = arith.muli %arg12, %c16_i32 : i32
        %131 = arith.index_cast %130 : i32 to index
        tpu.vector_store %arg7[%131], %129 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %132 = vector.extract %129[15] : i32 from vector<16xi32>
        scf.yield %132 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %77 = arith.subi %76, %72 : i32
      %78 = scf.for %arg12 = %c0_i32 to %75 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = vector.broadcast %77 : i32 to vector<16xi32>
        %128 = arith.cmpi sle, %126, %127 : vector<16xi32>
        %129 = tpu.all_reduce %128 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %130 = vector.extract %129[0] : i32 from vector<16xi32>
        %131 = arith.addi %arg13, %130 : i32
        scf.yield %131 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %79 = arith.index_cast %78 : i32 to index
      %80 = tpu.vector_load %arg7[%79] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %81 = vector.extract %80[0] : i32 from vector<16xi32>
      %82 = arith.cmpi sgt, %78, %c0_i32 : i32
      %83 = arith.subi %78, %c1_i32 : i32
      %84 = arith.maxsi %83, %c0_i32 : i32
      %85 = arith.index_cast %84 : i32 to index
      %86 = tpu.vector_load %arg7[%85] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %87 = vector.extract %86[0] : i32 from vector<16xi32>
      %88 = arith.select %82, %87, %c0_i32 : i32
      %89 = arith.subi %76, %81 : i32
      %90 = arith.subi %81, %88 : i32
      %91 = arith.subi %72, %89 : i32
      %92 = arith.subi %71, %c8_i32 : i32
      %93 = arith.shli %70, %92 : i32
      %94 = arith.ori %93, %78 : i32
      %95 = arith.cmpi eq, %91, %90 : i32
      %96 = arith.ori %69, %95 : i1
      %97 = arith.cmpi eq, %91, %c0_i32 : i32
      %98 = arith.ori %96, %97 : i1
      %99 = arith.select %69, %70, %94 : i32
      %100 = arith.select %69, %71, %c8_i32 : i32
      %101 = arith.select %69, %72, %91 : i32
      %102 = arith.select %98, %c0_i32, %c256_i32 : i32
      scf.for %arg12 = %c0_i32 to %102 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg6[%125], %cst_9 {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %103 = arith.select %98, %c0_i32, %21 : i32
      scf.for %arg12 = %c0_i32 to %103 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg5[%125] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
        %127 = tpu.bitcast %126 : vector<16xf32> -> vector<16xi32>
        %128 = arith.muli %arg12, %c16_i32 : i32
        %129 = vector.broadcast %128 : i32 to vector<16xi32>
        %130 = arith.addi %129, %2 : vector<16xi32>
        %131 = vector.broadcast %19 : i32 to vector<16xi32>
        %132 = arith.cmpi slt, %130, %131 : vector<16xi32>
        %133 = arith.cmpi sgt, %127, %cst_8 : vector<16xi32>
        %134 = arith.andi %132, %133 : vector<16xi1>
        %135 = vector.broadcast %100 : i32 to vector<16xi32>
        %136 = arith.shrsi %127, %135 : vector<16xi32>
        %137 = vector.broadcast %99 : i32 to vector<16xi32>
        %138 = arith.cmpi eq, %136, %137 : vector<16xi32>
        %139 = arith.andi %134, %138 : vector<16xi1>
        %140 = arith.andi %127, %cst_1 : vector<16xi32>
        %141 = arith.muli %2, %cst_5 : vector<16xi32>
        %142 = arith.addi %141, %140 : vector<16xi32>
        tpu.vector_store_idx %arg6[%142], %cst_4 masked %139 {add = true} : memref<4112xi32, #tpu.memory_space<vmem>>[vector<16xi32>], vector<16xi32>, vector<16xi1>
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %104 = arith.select %98, %c0_i32, %c16_i32 : i32
      scf.for %arg12 = %c0_i32 to %104 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg6[%125] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = arith.addi %124, %c256_i32 : i32
        %128 = arith.index_cast %127 : i32 to index
        %129 = tpu.vector_load %arg6[%128] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %130 = arith.addi %126, %129 : vector<16xi32>
        %131 = arith.addi %124, %c512_i32 : i32
        %132 = arith.index_cast %131 : i32 to index
        %133 = tpu.vector_load %arg6[%132] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %134 = arith.addi %130, %133 : vector<16xi32>
        %135 = arith.addi %124, %c768_i32 : i32
        %136 = arith.index_cast %135 : i32 to index
        %137 = tpu.vector_load %arg6[%136] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %138 = arith.addi %134, %137 : vector<16xi32>
        %139 = arith.addi %124, %c1024_i32 : i32
        %140 = arith.index_cast %139 : i32 to index
        %141 = tpu.vector_load %arg6[%140] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %142 = arith.addi %138, %141 : vector<16xi32>
        %143 = arith.addi %124, %c1280_i32 : i32
        %144 = arith.index_cast %143 : i32 to index
        %145 = tpu.vector_load %arg6[%144] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %146 = arith.addi %142, %145 : vector<16xi32>
        %147 = arith.addi %124, %c1536_i32 : i32
        %148 = arith.index_cast %147 : i32 to index
        %149 = tpu.vector_load %arg6[%148] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %150 = arith.addi %146, %149 : vector<16xi32>
        %151 = arith.addi %124, %c1792_i32 : i32
        %152 = arith.index_cast %151 : i32 to index
        %153 = tpu.vector_load %arg6[%152] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %154 = arith.addi %150, %153 : vector<16xi32>
        %155 = arith.addi %124, %c2048_i32 : i32
        %156 = arith.index_cast %155 : i32 to index
        %157 = tpu.vector_load %arg6[%156] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %158 = arith.addi %154, %157 : vector<16xi32>
        %159 = arith.addi %124, %c2304_i32 : i32
        %160 = arith.index_cast %159 : i32 to index
        %161 = tpu.vector_load %arg6[%160] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %162 = arith.addi %158, %161 : vector<16xi32>
        %163 = arith.addi %124, %c2560_i32 : i32
        %164 = arith.index_cast %163 : i32 to index
        %165 = tpu.vector_load %arg6[%164] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %166 = arith.addi %162, %165 : vector<16xi32>
        %167 = arith.addi %124, %c2816_i32 : i32
        %168 = arith.index_cast %167 : i32 to index
        %169 = tpu.vector_load %arg6[%168] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %170 = arith.addi %166, %169 : vector<16xi32>
        %171 = arith.addi %124, %c3072_i32 : i32
        %172 = arith.index_cast %171 : i32 to index
        %173 = tpu.vector_load %arg6[%172] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %174 = arith.addi %170, %173 : vector<16xi32>
        %175 = arith.addi %124, %c3328_i32 : i32
        %176 = arith.index_cast %175 : i32 to index
        %177 = tpu.vector_load %arg6[%176] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %178 = arith.addi %174, %177 : vector<16xi32>
        %179 = arith.addi %124, %c3584_i32 : i32
        %180 = arith.index_cast %179 : i32 to index
        %181 = tpu.vector_load %arg6[%180] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %182 = arith.addi %178, %181 : vector<16xi32>
        %183 = arith.addi %124, %c3840_i32 : i32
        %184 = arith.index_cast %183 : i32 to index
        %185 = tpu.vector_load %arg6[%184] {strides = array<i32>} : memref<4112xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %186 = arith.addi %182, %185 : vector<16xi32>
        %187 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg7[%187], %186 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %105 = scf.for %arg12 = %c0_i32 to %104 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = tpu.scan <sum>, %126 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %128 = vector.broadcast %arg13 : i32 to vector<16xi32>
        %129 = arith.addi %127, %128 : vector<16xi32>
        %130 = arith.muli %arg12, %c16_i32 : i32
        %131 = arith.index_cast %130 : i32 to index
        tpu.vector_store %arg7[%131], %129 {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %132 = vector.extract %129[15] : i32 from vector<16xi32>
        scf.yield %132 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %106 = arith.subi %105, %101 : i32
      %107 = scf.for %arg12 = %c0_i32 to %104 step %c1_i32 iter_args(%arg13 = %c0_i32) -> (i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg7[%125] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
        %127 = vector.broadcast %106 : i32 to vector<16xi32>
        %128 = arith.cmpi sle, %126, %127 : vector<16xi32>
        %129 = tpu.all_reduce %128 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %130 = vector.extract %129[0] : i32 from vector<16xi32>
        %131 = arith.addi %arg13, %130 : i32
        scf.yield %131 : i32
      } {sc.loop_unroll_factor = 2 : i64, sc.parallel_access}
      %108 = arith.index_cast %107 : i32 to index
      %109 = tpu.vector_load %arg7[%108] {strides = array<i32>} : memref<272xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      %110 = vector.extract %109[0] : i32 from vector<16xi32>
      %111 = arith.subi %105, %110 : i32
      %112 = arith.subi %101, %111 : i32
      %113 = arith.shli %99, %100 : i32
      %114 = arith.ori %113, %107 : i32
      %115 = arith.select %98, %99, %114 : i32
      %116 = arith.select %98, %100, %c0_i32 : i32
      %117 = arith.select %98, %101, %112 : i32
      %118 = arith.select %26, %c0_i32, %117 : i32
      %119 = arith.cmpi slt, %25, %c2048_i32 : i32
      %120 = arith.andi %9, %119 : i1
      %121 = arith.select %120, %c128_i32, %c0_i32 : i32
      scf.for %arg12 = %c0_i32 to %121 step %c1_i32  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        tpu.vector_store %arg8[%125], %cst {strides = array<i32>} : memref<2064xi32, #tpu.memory_space<vmem>>, vector<16xi32>, 
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      %122 = arith.select %9, %21, %c0_i32 : i32
      %123:2 = scf.for %arg12 = %c0_i32 to %122 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
        %124 = arith.muli %arg12, %c16_i32 : i32
        %125 = arith.index_cast %124 : i32 to index
        %126 = tpu.vector_load %arg5[%125] {strides = array<i32>} : memref<10240xf32, #tpu.memory_space<vmem>>, vector<16xf32>, 
        %127 = tpu.bitcast %126 : vector<16xf32> -> vector<16xi32>
        %128 = arith.muli %arg12, %c16_i32 : i32
        %129 = vector.broadcast %128 : i32 to vector<16xi32>
        %130 = arith.addi %129, %2 : vector<16xi32>
        %131 = vector.broadcast %19 : i32 to vector<16xi32>
        %132 = arith.cmpi slt, %130, %131 : vector<16xi32>
        %133 = arith.cmpi sgt, %127, %cst_8 : vector<16xi32>
        %134 = arith.andi %132, %133 : vector<16xi1>
        %135 = vector.broadcast %116 : i32 to vector<16xi32>
        %136 = arith.shrsi %127, %135 : vector<16xi32>
        %137 = vector.broadcast %115 : i32 to vector<16xi32>
        %138 = arith.cmpi sgt, %136, %137 : vector<16xi32>
        %139 = arith.andi %134, %138 : vector<16xi1>
        %140 = vector.broadcast %115 : i32 to vector<16xi32>
        %141 = arith.cmpi eq, %136, %140 : vector<16xi32>
        %142 = arith.andi %134, %141 : vector<16xi1>
        %143 = arith.extui %142 : vector<16xi1> to vector<16xi32>
        %144 = tpu.scan <sum>, %143 masked %cst_3 : vector<16xi32>, vector<16xi1> -> vector<16xi32>
        %145 = vector.broadcast %arg14 : i32 to vector<16xi32>
        %146 = arith.addi %144, %145 : vector<16xi32>
        %147 = vector.broadcast %118 : i32 to vector<16xi32>
        %148 = arith.cmpi sle, %146, %147 : vector<16xi32>
        %149 = arith.andi %142, %148 : vector<16xi1>
        %150 = arith.ori %139, %149 : vector<16xi1>
        %151 = arith.select %26, %134, %150 : vector<16xi1>
        %152 = arith.index_cast %arg13 : i32 to index
        tpu.vector_store %arg8[%152], %130 masked %151 {strides = array<i32>} : memref<2064xi32, #tpu.memory_space<vmem>>, vector<16xi32>, vector<16xi1>
        %153 = tpu.all_reduce %151 {dim = 0 : i64, kind = #tpu.reduction_kind<sum>} : vector<16xi1> -> vector<16xi32>
        %154 = vector.extract %153[0] : i32 from vector<16xi32>
        %155 = arith.addi %arg13, %154 : i32
        %156 = vector.extract %146[15] : i32 from vector<16xi32>
        scf.yield %155, %156 : i32, i32
      } {sc.loop_unroll_factor = 4 : i64, sc.parallel_access}
      scf.if %9 {
        %124 = arith.muli %10, %c2048_i32 : i32
        %125 = tpu.memref_slice %arg10[%c2_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %126 = tpu.memref_squeeze %125 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %127 = tpu.memref_slice %arg4[%124] : memref<2097152xi32, #tpu.memory_space<hbm>> -> memref<2048xi32, #tpu.memory_space<hbm>>
        %128 = tpu.memref_slice %arg8[%c0_i32] : memref<2064xi32, #tpu.memory_space<vmem>> -> memref<2048xi32, #tpu.memory_space<vmem>>
        tpu.enqueue_dma source(%128 : memref<2048xi32, #tpu.memory_space<vmem>>) target(%127 : memref<2048xi32, #tpu.memory_space<hbm>>) target_semaphore(%126 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
        %129 = tpu.memref_slice %arg10[%c2_i32] : memref<4x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %130 = tpu.memref_squeeze %129 : memref<1x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>> -> memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>
        %131 = tpu.memref_slice %arg4[%124] : memref<2097152xi32, #tpu.memory_space<hbm>> -> memref<2048xi32, #tpu.memory_space<hbm>>
        %132 = tpu.memref_slice %arg8[%c0_i32] : memref<2064xi32, #tpu.memory_space<vmem>> -> memref<2048xi32, #tpu.memory_space<vmem>>
        tpu.wait_dma2 semaphore(%130 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%132 : memref<2048xi32, #tpu.memory_space<vmem>>) dst(%131 : memref<2048xi32, #tpu.memory_space<hbm>>)
      }
    }
    return
  }
}

