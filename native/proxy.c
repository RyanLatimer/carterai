// Resolve the effective route with CFNetwork, including exception lists and PAC/WPAD.
// Called only from the networking worker. Each PAC evaluation has its own run loop.
#include <CoreFoundation/CoreFoundation.h>
#include <CFNetwork/CFNetwork.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

typedef struct { CFArrayRef proxies; CFErrorRef error; int done; } Evaluation;
static void evaluated(void *info, CFArrayRef proxies, CFErrorRef error) {
    Evaluation *state = info;
    if (proxies) state->proxies = CFRetain(proxies);
    if (error) state->error = CFRetain(error);
    state->done = 1;
}
static int string(CFStringRef value, char *out, size_t cap) {
    return value && CFGetTypeID(value) == CFStringGetTypeID() &&
        CFStringGetCString(value, out, cap, kCFStringEncodingUTF8);
}
static int resolve(CFArrayRef proxies, CFURLRef target, char *out, size_t cap, int depth) {
    if (!proxies || depth > 3 || CFArrayGetCount(proxies) == 0) {
        snprintf(out, cap, "System proxy resolution returned no route"); return -1;
    }
    for (CFIndex i = 0; i < CFArrayGetCount(proxies); i++) {
        CFDictionaryRef proxy = CFArrayGetValueAtIndex(proxies, i);
        CFStringRef type = CFDictionaryGetValue(proxy, kCFProxyTypeKey);
        if (type && CFEqual(type, kCFProxyTypeNone)) { out[0] = 0; return 0; }
        if (type && (CFEqual(type,kCFProxyTypeAutoConfigurationURL) || CFEqual(type,kCFProxyTypeAutoConfigurationJavaScript))) {
            Evaluation state = {0};
            CFStreamClientContext context = {0, &state, NULL, NULL, NULL};
            CFRunLoopSourceRef source = NULL;
            if (CFEqual(type,kCFProxyTypeAutoConfigurationURL)) {
                CFURLRef url = CFDictionaryGetValue(proxy,kCFProxyAutoConfigurationURLKey);
                if (url) source = CFNetworkExecuteProxyAutoConfigurationURL(url,target,evaluated,&context);
            } else {
                CFStringRef script = CFDictionaryGetValue(proxy,kCFProxyAutoConfigurationJavaScriptKey);
                if (script) source = CFNetworkExecuteProxyAutoConfigurationScript(script,target,evaluated,&context);
            }
            if (!source) { snprintf(out,cap,"Cannot start system PAC evaluation"); return -1; }
            CFRunLoopAddSource(CFRunLoopGetCurrent(),source,kCFRunLoopDefaultMode);
            CFAbsoluteTime until = CFAbsoluteTimeGetCurrent() + 15.0;
            while (!state.done && CFAbsoluteTimeGetCurrent() < until)
                CFRunLoopRunInMode(kCFRunLoopDefaultMode,0.1,true);
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(),source,kCFRunLoopDefaultMode);
            CFRunLoopSourceInvalidate(source); CFRelease(source);
            int result;
            if (!state.done || state.error || !state.proxies) {
                char detail[1024] = "PAC timed out";
                if (state.error) {
                    CFStringRef description = CFErrorCopyDescription(state.error);
                    string(description,detail,sizeof(detail));
                    if (description) CFRelease(description);
                }
                snprintf(out,cap,"System PAC failed: %s",detail); result = -1;
            } else result = resolve(state.proxies,target,out,cap,depth+1);
            if (state.proxies) CFRelease(state.proxies);
            if (state.error) CFRelease(state.error);
            return result;
        }
        const char *scheme = NULL;
        if (type && CFEqual(type,kCFProxyTypeHTTP)) scheme = "http";
        else if (type && CFEqual(type,kCFProxyTypeHTTPS)) scheme = "https";
        else if (type && CFEqual(type,kCFProxyTypeSOCKS)) scheme = "socks5h";
        if (!scheme) continue;
        char host[2048]; int port = 0;
        CFStringRef name = CFDictionaryGetValue(proxy,kCFProxyHostNameKey);
        CFNumberRef number = CFDictionaryGetValue(proxy,kCFProxyPortNumberKey);
        if (!string(name,host,sizeof(host)) || !number || !CFNumberGetValue(number,kCFNumberIntType,&port) || port <= 0 || port > 65535) {
            snprintf(out,cap,"Invalid system proxy host or port"); return -1;
        }
        int n = snprintf(out,cap,strchr(host,':') ? "%s://[%s]:%d" : "%s://%s:%d",scheme,host,port);
        if (n < 0 || (size_t)n >= cap) { snprintf(out,cap,"System proxy address too long"); return -1; }
        return 1;
    }
    snprintf(out,cap,"Unsupported system proxy route"); return -1;
}
// Returns 0 for DIRECT, 1 for a proxy URI, -1 with diagnostic text on failure.
int carter_proxy_for_url(const char *url_string, char *out, size_t cap) {
    if (!out || cap < 128 || !url_string) return -1;
    CFStringRef input = CFStringCreateWithCString(NULL,url_string,kCFStringEncodingUTF8);
    CFURLRef url = input ? CFURLCreateWithString(NULL,input,NULL) : NULL;
    if (input) CFRelease(input);
    if (!url) { snprintf(out,cap,"Invalid URL for proxy resolution"); return -1; }
    CFDictionaryRef settings = CFNetworkCopySystemProxySettings();
    if (!settings) { CFRelease(url); snprintf(out,cap,"Cannot read macOS system proxies"); return -1; }
    CFArrayRef proxies = CFNetworkCopyProxiesForURL(url,settings);
    int result = resolve(proxies,url,out,cap,0);
    if (proxies) CFRelease(proxies);
    CFRelease(settings); CFRelease(url);
    return result;
}
