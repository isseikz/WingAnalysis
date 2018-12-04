function[R0] = Foo(x,x_est,cutOffFreq) // [Hz]
    close()
    
    samplerate = 300;
    dt = 1/ samplerate;
    alpha = dt * cutOffFreq / (1+dt*cutOffFreq)
    
    deg2Rot = 360;
    [sx, sy] = size(x);
    f = zeros(sx,sy);
    f(1,:) = x(1,:);
    for i = 1:sx-1
        f(i+1,:) = f(i,:) * (1-alpha) + x(i+1,:) * alpha;
    end
    filtered = f;
    
    wc= alpha/(1-alpha) * dt;
    disp("cut off freq:",wc, "\r\n","Period:", 1/wc)
    
    a = sqrt(f(:,2).^2+f(:,3).^2+f(:,4).^2);
    g = sqrt(f(:,5).^2+f(:,6).^2+f(:,7).^2)/deg2Rot;
    
    t_est =  x_est(:,1) + 0.1
    a_est = -x_est(:,2);
    g_est = -x_est(:,3);
    
    clf;
    plt1 = plot(x(:,1),f(:,2),x(:,1),f(:,3),x(:,1),f(:,4),x(:,1),f(:,5)/deg2Rot,x(:,1),f(:,6)/deg2Rot,x(:,1),f(:,7)/deg2Rot);
    legend(['ax[G]','ay[G]','az[G]','gx[rps]','gy[rps]','gz[rps]']);
    
    scf();
    plt2 = plot(x(:,1)-x(1,1),a(:),'r',x(:,1)-x(1,1),g(:),'b');
    plt2 = plot(t_est(:,1)+0.1,a_est(:),'r-',t_est(:,1)+0.1,g_est(:),'b-');
    xlabel('time[s]')
    graph = gca();
    graph.font_size=3;
    graph.x_label.font_size=4
    legend(['Acceleration [G]','Angular Velocity [rps]'],'in_upper_left');
    
    scf();
    plt2 = plot(x(:,1)-x(1,1),x(:,2),'r',x(:,1)-x(1,1),x(:,5)/deg2Rot,'b');
    plt2 = plot(t_est(:,1),a_est(:),'r:',t_est(:,1),g_est(:),'b:');
    xlabel('time[s]')
    graph = gca();
    graph.font_size=3;
    graph.x_label.font_size=4
    legend(['Acceleration [G]','Angular Velocity [rps]'],'in_upper_left');
    
    
    ax0 = mean(x(1:30,2));
    ay0 = mean(x(1:30,3));
    az0 = mean(x(1:30,4));
    sin_phi =  ay0;
    cos_phi = sqrt(1-sin_phi.^2);
    sin_gma = -ax0 / cos_phi;
    cos_gma =  az0 / cos_phi;
    Rx0 = [
        1, 0,        0;
        0, cos_phi, -sin_phi;
        0, sin_phi,  cos_phi; 
    ];
    Ry0 = [
         cos_gma, 0, sin_gma;
         0,       1, 0;
        -sin_gma, 0, cos_gma; 
    ];
    R0 = (Rx0 * Ry0)';
endfunction
